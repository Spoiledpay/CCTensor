"""Treino de conversa em ingles do cotidiano (dataset input/chat_en.jsonl).

Terceira geracao do pipeline: SEM alfabeto, SEM JSON no prompt. O modelo
aprende o template

    <bos>User: ...\\nAssistant: ...\\n<eos>

e a loss e calculada SO nas respostas (prompt_weight=0), o que evita o modelo
gastar capacidade copiando o que ja esta no contexto.

Metricas por epoca:
  - loss de treino (por token de resposta)
  - loss de validacao em INTENCOES NAO VISTAS (split disjunto por id)
  - exact-match multi-referencia (aceita qualquer variante da resposta)

Uso:
    .\\.venv\\Scripts\\python.exe train_chat.py
    .\\.venv\\Scripts\\python.exe train_chat.py --epochs 10 --out modelos/chat_en.pt
"""

import argparse
import csv
import math
import os
import random
import sys
import time

import torch

from minigpt import chat_dataset as cd
from minigpt.config import DEFAULT_CONFIG, load_config, section
from minigpt.model import TinyGPT
from minigpt.tokenizer import ByteTokenizer
from minigpt.utils import save_checkpoint


def make_length_batches(lengths, batch_size, seed, epoch):
    """Lotes com comprimentos parecidos: corta o padding -> passos mais rapidos."""
    rng = random.Random(seed + epoch)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    idx.sort(key=lambda i: lengths[i])
    batches = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    rng.shuffle(batches)
    return batches


def make_scheduler(opt, total, warmup):
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=DEFAULT_CONFIG)
    known, _ = pre.parse_known_args(argv)
    cfg = section(load_config(known.config), "chat")

    p = argparse.ArgumentParser(description="Treino de conversa (chat_en)")
    p.add_argument("--config", default=known.config)
    p.add_argument("--dataset", default=cfg.get("dataset", "input/chat_en.jsonl"))
    p.add_argument("--out", default=cfg.get("out", "modelos/chat_en.pt"))
    p.add_argument("--epochs", type=int, default=cfg.get("epochs", 10))
    p.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 64))
    p.add_argument("--block-size", type=int, default=cfg.get("block_size", 192))
    p.add_argument("--n-embd", type=int, default=cfg.get("n_embd", 288))
    p.add_argument("--n-head", type=int, default=cfg.get("n_head", 8))
    p.add_argument("--n-layer", type=int, default=cfg.get("n_layer", 6))
    p.add_argument("--lr", type=float, default=cfg.get("lr", 3e-3))
    p.add_argument("--weight-decay", type=float, default=cfg.get("weight_decay", 0.01))
    p.add_argument("--answer-weight", type=float, default=cfg.get("answer_weight", 20.0))
    p.add_argument("--stop-weight", type=float, default=cfg.get("stop_weight", 1.0))
    p.add_argument("--prompt-weight", type=float, default=cfg.get("prompt_weight", 0.0))
    p.add_argument("--n-per-entry", type=int, default=cfg.get("n_per_entry", 20),
                   help="conversas geradas por intencao no treino")
    p.add_argument("--val-frac", type=float, default=cfg.get("val_frac", 0.12))
    p.add_argument("--seed", type=int, default=cfg.get("seed", 1337))
    p.add_argument("--num-threads", type=int, default=cfg.get("num_threads", 8))
    p.add_argument("--eval-probe", type=int, default=cfg.get("eval_probe", 2),
                   help="geracoes de exact-match por intencao de validacao")
    p.add_argument("--resume", nargs="?", const="auto", default=None)
    p.add_argument("--no-onnx", action="store_true")
    return p.parse_args(argv)


def export_onnx(path, no_onnx):
    if no_onnx:
        return
    try:
        from export_onnx import export_checkpoint_to_onnx
        out = os.path.splitext(path)[0] + ".onnx"
        export_checkpoint_to_onnx(path, out)
        print(f"ONNX: {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"aviso: falha ao exportar ONNX: {exc}")


def main(argv=None):
    args = parse_args(argv)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    tok = ByteTokenizer()
    print(f"Dataset: {args.dataset}")
    n_e, n_t, topics, n_ut, n_at = cd.validate(args.dataset)
    print(f"  {n_e} intencoes | {n_t} turnos | {len(topics)} topicos | "
          f"{n_ut} variantes de usuario | {n_at} de assistente")

    data = cd.build_datasets(
        tokenizer=tok, block_size=args.block_size, n_per_entry=args.n_per_entry,
        val_frac=args.val_frac, seed=args.seed,
        answer_weight=args.answer_weight, stop_weight=args.stop_weight,
        prompt_weight=args.prompt_weight, path=args.dataset)
    st, sv = data["train_stats"], data["val_stats"]
    print(f"  treino: {st['n']} conversas de {st['n_entries']} intencoes "
          f"(len medio {st['mean_len']:.0f}, max {st['max_len']})")
    print(f"  val   : {sv['n']} conversas de {sv['n_entries']} intencoes INEDITAS "
          f"(len medio {sv['mean_len']:.0f}, max {sv['max_len']})")
    if st["skipped_too_long"] or sv["skipped_too_long"]:
        print(f"  aviso: {st['skipped_too_long'] + sv['skipped_too_long']} "
              f"conversas nao couberam em block_size={args.block_size}")

    x, y, w = data["x"], data["y"], data["w"]
    xv, yv, wv = data["xv"], data["yv"], data["wv"]
    lengths = (x != tok.pad_id).sum(dim=1).tolist()

    steps_per_epoch = (x.shape[0] + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    torch.manual_seed(args.seed)
    model = TinyGPT(tok.vocab_size, block_size=args.block_size, n_embd=args.n_embd,
                    n_head=args.n_head, n_layer=args.n_layer, tie_embeddings=True)
    print(f"  modelo: {model.num_params:,} parametros | "
          f"{args.epochs} epocas x {steps_per_epoch} passos = {total_steps} passos")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = make_scheduler(opt, total_steps, min(200, max(10, total_steps // 10)))

    start_epoch = 1
    if args.resume:
        path = args.out if args.resume == "auto" else args.resume
        if os.path.exists(path):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            if ck.get("optimizer"):
                opt.load_state_dict(ck["optimizer"])
            if ck.get("scheduler"):
                sched.load_state_dict(ck["scheduler"])
            start_epoch = int(ck.get("epoch", 0)) + 1
            print(f"  RESUME de {path}: epoca {start_epoch}/{args.epochs}")
        else:
            print(f"  aviso: --resume sem arquivo {path}; comecando do zero")

    config = {
        "task": "chat_en",
        "dataset": args.dataset,
        "block_size": args.block_size,
        "n_embd": args.n_embd, "n_head": args.n_head, "n_layer": args.n_layer,
        "dropout": 0.0, "tie_embeddings": True, "pos_encoding": "learned",
        "copy_head": False, "normalize_embeddings": False,
        "vocab_size": tok.vocab_size,
        "max_new_tokens": 48,
        "answer_weight": args.answer_weight,
        "stop_weight": args.stop_weight,
        "prompt_weight": args.prompt_weight,
        "topics": data["topics"],
        "train_entries": data["train_entries"],
        "val_entries": data["val_entries"],
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(args.out)[0]
    hist_path = stem + "_history.csv"
    keep = start_epoch > 1 and os.path.exists(hist_path)
    hist = open(hist_path, "a" if keep else "w", encoding="utf-8", newline="")
    hw = csv.writer(hist)
    if not keep:
        hw.writerow(["epoch", "loss", "val_loss", "val_ppl", "probe", "n_probe",
                     "lr", "secs"])

    probe_items = cd.probe_items(
        [e for e in cd.load_catalog(args.dataset) if e["id"] in set(data["val_entries"])],
        tok, per_entry=args.eval_probe, seed=args.seed + 3)

    def val_loss():
        model.eval()
        tot, nb = 0.0, 0
        with torch.no_grad():
            vlen = (xv != tok.pad_id).sum(dim=1).tolist()
            for bidx in make_length_batches(vlen, args.batch_size, args.seed, 0):
                T = max(vlen[i] for i in bidx)
                _, _, l = model(xv[bidx][:, :T], targets=yv[bidx][:, :T],
                                weights=wv[bidx][:, :T])
                tot += float(l)
                nb += 1
        model.train()
        return tot / max(1, nb)

    best = float("inf")
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        ep_t0 = time.time()
        bl = make_length_batches(lengths, args.batch_size, args.seed, epoch)
        loss_sum, nb, wsum, wtot = 0.0, 0, 0.0, 0.0
        for b, bidx in enumerate(bl, start=1):
            T = max(lengths[i] for i in bidx)
            bx, by, bw = x[bidx][:, :T], y[bidx][:, :T], w[bidx][:, :T]
            _, _, loss = model(bx, targets=by, weights=bw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += float(loss.detach())
            nb += 1
            if b % max(1, steps_per_epoch // 4) == 0 or b == steps_per_epoch:
                el = time.time() - ep_t0
                bps = b / max(el, 1e-6)
                eta = (steps_per_epoch - b) / bps + (args.epochs - epoch) * (el / b * steps_per_epoch)
                print(f"    epoca {epoch}/{args.epochs} batch {b}/{steps_per_epoch} "
                      f"loss {float(loss):.4f} {bps:.1f} batch/s eta_total {eta/60:.0f}min",
                      flush=True)

        tr_loss = loss_sum / max(1, nb)
        vl = val_loss()
        ok, npr, per_topic = cd.evaluate_probe(model, tok, probe_items)
        acc = ok / max(1, npr)
        lr_now = sched.get_last_lr()[0]
        print(f"  epoca {epoch:3d}/{args.epochs}  loss {tr_loss:.4f}  "
              f"val {vl:.4f} (ppl {math.exp(min(vl, 20)):.2f})  "
              f"probe {ok}/{npr} = {acc:.1%}  lr {lr_now:.2e}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        hw.writerow([epoch, f"{tr_loss:.4f}", f"{vl:.4f}",
                     f"{math.exp(min(vl, 20)):.2f}", f"{acc:.4f}", npr,
                     f"{lr_now:.2e}", f"{time.time()-t0:.0f}"])
        hist.flush()

        if vl < best:
            best = vl
            save_checkpoint(args.out, model, tok, config,
                            extra={"epoch": epoch, "optimizer": opt.state_dict(),
                                   "scheduler": sched.state_dict()})
            print(f"    -> melhor val_loss ate agora ({vl:.4f}); checkpoint salvo")

    hist.close()
    if not os.path.exists(args.out):
        save_checkpoint(args.out, model, tok, config,
                        extra={"epoch": args.epochs, "optimizer": opt.state_dict(),
                               "scheduler": sched.state_dict()})
    print(f"\nModelo: {args.out}")
    print(f"Historico: {hist_path}")
    print(f"Melhor val_loss: {best:.4f}")
    export_onnx(args.out, args.no_onnx)
    return 0


if __name__ == "__main__":
    sys.exit(main())

