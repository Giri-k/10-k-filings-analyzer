import gradio as gr
from agent import call_agent

CSS = """
.header { text-align: center; margin-bottom: 0.5em; }
.header h1 { margin-bottom: 0.1em; }
.header p { color: #666; font-size: 1.05em; }
.status-box textarea { font-family: monospace; font-size: 0.9em; }
.output-box { min-height: 400px; }
.examples-row { margin-top: 0.5em; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css=CSS,
    title="10-K Filings Analyzer",
) as demo:

    gr.HTML("""
    <div class="header">
        <h1>10-K Filings Analyzer</h1>
        <p>AI-powered analysis of SEC 10-K filings using hybrid retrieval and LLM generation</p>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            symbol = gr.Textbox(
                label="Ticker Symbol",
                placeholder="e.g., AAPL, MSFT, GOOGL",
                max_lines=1,
            )
            query = gr.Textbox(
                label="Question",
                placeholder="What are the key risk factors?",
                lines=3,
            )
            submit = gr.Button("Analyze", variant="primary", size="lg")
            clear = gr.ClearButton(
                components=[symbol, query],
                value="Clear",
                size="sm",
            )
            status = gr.Textbox(
                label="Pipeline Status",
                interactive=False,
                lines=1,
                max_lines=2,
                elem_classes=["status-box"],
            )
            with gr.Accordion("Example queries", open=False, elem_classes=["examples-row"]):
                gr.Examples(
                    examples=[
                        ["AAPL", "What are Apple's major risk factors?"],
                        ["AAPL", "What does management discuss about future outlook?"],
                        ["MSFT", "What are the key financial trends?"],
                        ["", "Compare risk factors between AAPL and MSFT"],
                    ],
                    inputs=[symbol, query],
                )

        with gr.Column(scale=2, min_width=500):
            output = gr.Markdown(
                value="*Submit a ticker and question to get started.*",
                elem_classes=["output-box"],
            )

    submit.click(
        fn=call_agent,
        inputs=[symbol, query],
        outputs=[status, output],
    )

demo.launch()
