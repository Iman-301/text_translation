from __future__ import annotations

import gradio as gr

from translator import TranslationConfig, TranslationError, translate_text


def run_translation(text: str, src: str, tgt: str, max_length: int) -> str:
    try:
        config = TranslationConfig(src_lang=src, tgt_lang=tgt, max_length=max_length)
        return translate_text(text, config)
    except TranslationError as err:
        return f"Error: {err}"


with gr.Blocks(title="Cognitive Science Text Translator") as demo:
    gr.Markdown("# Cognitive Science Text Translator")
    gr.Markdown(
        "Translate text between languages using pretrained neural MT models. "
        "Useful for multilingual cognitive science material."
    )

    with gr.Row():
        src_lang = gr.Textbox(value="en", label="Source language code")
        tgt_lang = gr.Textbox(value="ar", label="Target language code")

    input_text = gr.Textbox(lines=6, label="Input text")
    max_length = gr.Slider(minimum=32, maximum=512, value=256, step=16, label="Max length")
    translate_btn = gr.Button("Translate")
    output_text = gr.Textbox(lines=6, label="Translated text")

    translate_btn.click(
        fn=run_translation,
        inputs=[input_text, src_lang, tgt_lang, max_length],
        outputs=output_text,
    )


if __name__ == "__main__":
    demo.launch()
