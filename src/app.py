from __future__ import annotations

from pathlib import Path

import gradio as gr

from seq2seq_backend import Seq2SeqBackendError, Seq2SeqConfig, translate_with_seq2seq


def run_translation(
    target_lang: str,
    text: str,
    max_length: int,
    scratch_method: str,
    scratch_beam_width: int,
    checkpoint_path: str,
    vocab_dir: str,
    device: str,
) -> tuple[str, str]:
    try:
        root = Path(__file__).resolve().parent.parent
        scratch_cfg = Seq2SeqConfig(
            checkpoint_path=(root / checkpoint_path).resolve()
            if not Path(checkpoint_path).is_absolute()
            else Path(checkpoint_path),
            vocab_dir=(root / vocab_dir).resolve() if not Path(vocab_dir).is_absolute() else Path(vocab_dir),
            target_lang=target_lang,
            method="beam_search" if scratch_method == "Beam search" else "greedy",
            beam_width=scratch_beam_width,
            max_length=min(int(max_length), 80),
            device="cuda" if device == "cuda" else "cpu",
        )
        out, meta = translate_with_seq2seq(text, scratch_cfg)
        return out, meta
    except (Seq2SeqBackendError, ValueError) as err:
        return f"Error: {err}", ""


with gr.Blocks(title="Cognitive Science Scratch Translator (Seq2Seq + Attention)") as demo:
    gr.Markdown("# Cognitive Science Scratch Translator (Seq2Seq + Attention)")
    gr.Markdown(
        "This is a **from-scratch** neural machine translation demo.\n\n"
        "**Cognitive science angle**:\n"
        "- Encoder hidden state ≈ working-memory encoding\n"
        "- Attention weights ≈ selective attention during translation\n"
        "- Decoder generation ≈ incremental language production"
    )

    with gr.Row():
        checkpoint_path = gr.Textbox(
            value="seq2seq/models/checkpoints/best_model.pt",
            label="Checkpoint path",
        )
        vocab_dir = gr.Textbox(
            value="seq2seq/data/vocab",
            label="Vocab dir",
        )
        target_lang = gr.Dropdown(
            choices=["fr", "es", "de"],
            value="fr",
            label="Target language",
        )

    input_text = gr.Textbox(lines=6, label="Input text (English)")
    max_length = gr.Slider(minimum=16, maximum=128, value=50, step=1, label="Max length")

    with gr.Row():
        scratch_method = gr.Dropdown(
            choices=["Greedy", "Beam search"],
            value="Greedy",
            label="Decoding method",
        )
        scratch_beam_width = gr.Slider(
            minimum=2,
            maximum=8,
            value=5,
            step=1,
            label="Beam width (beam search)",
        )
        device = gr.Dropdown(choices=["cpu", "cuda"], value="cpu", label="Device")

    translate_btn = gr.Button("Translate")
    output_text = gr.Textbox(lines=6, label="Translated text")
    model_info = gr.Textbox(lines=1, label="Model info", interactive=False)

    translate_btn.click(
        fn=run_translation,
        inputs=[
            target_lang,
            input_text,
            max_length,
            scratch_method,
            scratch_beam_width,
            checkpoint_path,
            vocab_dir,
            device,
        ],
        outputs=[output_text, model_info],
    )


if __name__ == "__main__":
    demo.launch()
