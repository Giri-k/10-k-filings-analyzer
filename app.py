import gradio as gr
from agent import call_agent

with gr.Blocks() as demo:
    gr.Markdown("10K Filings Analyzer")
    with gr.Row():
        with gr.Column():
            symbol = gr.Textbox(label="Company Symbol")
            query = gr.Textbox(label="Question")
            submit = gr.Button("Submit")
        with gr.Column():
            output = gr.Textbox(label="Insights", lines=15, max_lines=30, show_copy_button=True)
    submit.click(fn=call_agent, inputs=[symbol, query], outputs=output)
    demo.launch()

# if __name__ == "__main__":
