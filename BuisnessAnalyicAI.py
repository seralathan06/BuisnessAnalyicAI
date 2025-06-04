import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import torch
from langchain_community.llms import HuggingFacePipeline

class BIAssistant:
    def __init__(self):
        self.data = None
        self.vectorstore = None
        self.qa_chain = None
        self.column_map = {
            'date': None,
            'product': None,
            'region': None,
            'sales': None,
            'customer_age': None
        }

    def load_data(self, file_path):
        """Load and prepare data"""
        try:
            self.data = pd.read_csv(file_path.name)
            self._detect_columns()

            if self.column_map['date']:
                self.data[self.column_map['date']] = pd.to_datetime(
                    self.data[self.column_map['date']],
                    errors='coerce'
                )

            return "Data loaded successfully!", self._get_data_preview()
        except Exception as e:
            return f"Error loading data: {str(e)}", None

    def _detect_columns(self):
        """Map expected columns to actual column names"""
        for col in self.data.columns:
            col_lower = col.lower()
            for expected_col in self.column_map.keys():
                if expected_col in col_lower:
                    self.column_map[expected_col] = col

    def _get_data_preview(self):
        """Generate HTML preview of data"""
        return f"""
        <h3>Data Preview</h3>
        <p>Rows: {len(self.data)}, Columns: {len(self.data.columns)}</p>
        {self.data.head().to_html()}
        """

    def initialize_system(self):
        """Initialize the knowledge base and LLM chain"""
        try:
            # Create knowledge base
            stats = self._generate_statistical_summaries()
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separator="\n"
            )
            docs = text_splitter.create_documents([stats])

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            self.vectorstore = FAISS.from_documents(docs, embeddings)

            # Initialize LLM
            model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side="left",
                pad_token="<|endoftext|>"
            )
            model = AutoModelForCausalLM.from_pretrained(model_name)

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=150,
                device=0 if torch.cuda.is_available() else -1,
                pad_token_id=tokenizer.eos_token_id
            )

            llm = HuggingFacePipeline(pipeline=pipe)

            prompt_template = """Answer concisely using ONLY this business data:
            {context}

            Question: {question}

            Answer in this format:
            [Summary] 1-2 sentence overview
            [Top Products] List top 3 if available
            [Key Metric] Most relevant number
            [Recommendation] One actionable suggestion

            Answer:"""

            PROMPT = ChatPromptTemplate.from_template(prompt_template)

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={
                    "prompt": PROMPT,
                    "document_variable_name": "context"
                }
            )

            return "System initialized successfully!"
        except Exception as e:
            return f"Initialization failed: {str(e)}"

    def _generate_statistical_summaries(self):
        """Generate concise statistical summaries"""
        summary = "Business Data Summary:\n"

        # Product Summary
        product_col = self.column_map['product']
        sales_col = self.column_map['sales']
        if product_col and sales_col:
            try:
                top_products = self.data.groupby(product_col)[sales_col].sum().nlargest(5)
                summary += f"\nTop 5 Products by Sales:\n{top_products.to_string()}\n"
            except:
                summary += "\nProduct analysis unavailable\n"

        # Sales Trends
        date_col = self.column_map['date']
        if date_col and sales_col:
            try:
                monthly_sales = self.data.groupby(pd.Grouper(key=date_col, freq='ME'))[sales_col].sum()
                summary += f"\nLast 3 Months Sales:\n{monthly_sales.tail(3).to_string()}\n"
            except:
                summary += "\nSales trend analysis unavailable\n"

        return summary

    def ask_question(self, question):
        """Get answer to business question"""
        try:
            if not self.qa_chain:
                return "System not initialized. Please load data and initialize first."

            result = self.qa_chain.invoke({"query": question})
            return result["result"]
        except Exception as e:
            return f"Error answering question: {str(e)}"

    def generate_plot(self, plot_type):
        """Generate data visualizations"""
        try:
            date_col = self.column_map['date']
            sales_col = self.column_map['sales']
            product_col = self.column_map['product']

            plt.figure(figsize=(10, 6))

            if plot_type == "Sales Trends" and date_col and sales_col:
                time_data = self.data.groupby(pd.Grouper(key=date_col, freq='ME'))[sales_col].sum()
                plt.plot(time_data.index, time_data.values)
                plt.title("Monthly Sales Trends")
                plt.xlabel("Month")
                plt.ylabel("Total Sales")
                plt.xticks(rotation=45)
                plt.tight_layout()
                return plt

            elif plot_type == "Product Performance" and product_col and sales_col:
                product_data = self.data.groupby(product_col)[sales_col].sum().nlargest(10)
                product_data.plot(kind='bar')
                plt.title("Top Products by Sales")
                plt.xlabel("Product")
                plt.ylabel("Total Sales")
                plt.tight_layout()
                return plt

        except Exception as e:
            print(f"Plot generation error: {str(e)}")
        return None
import gradio as gr
assistant = BIAssistant()

with gr.Blocks(title="Business Intelligence Assistant") as demo:
    gr.Markdown("# 🚀 AI-Powered Business Intelligence Assistant")

    with gr.Tab("Data Setup"):
        with gr.Row():
            data_file = gr.File(label="Upload CSV File", file_types=[".csv"])
            load_btn = gr.Button("Load Data")
        data_status = gr.Textbox(label="Status")
        data_preview = gr.HTML(label="Data Preview")
        init_btn = gr.Button("Initialize System")
        init_status = gr.Textbox(label="Initialization Status")

    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(label="Your business question", placeholder="E.g., What are our top products?")
        ask_btn = gr.Button("Ask")
        answer_output = gr.Textbox(label="Answer", lines=5)

    with gr.Tab("Visualizations"):
        plot_type = gr.Dropdown(
            label="Select Visualization",
            choices=["Sales Trends", "Product Performance"]
        )
        plot_btn = gr.Button("Generate Plot")
        plot_output = gr.Plot(label="Visualization")

    # Event handlers
    load_btn.click(
        assistant.load_data,
        inputs=[data_file],
        outputs=[data_status, data_preview]
    )

    init_btn.click(
        assistant.initialize_system,
        inputs=[],
        outputs=[init_status]
    )

    ask_btn.click(
        assistant.ask_question,
        inputs=[question_input],
        outputs=[answer_output]
    )

    plot_btn.click(
        assistant.generate_plot,
        inputs=[plot_type],
        outputs=[plot_output]
    )

if __name__ == "__main__":
    demo.launch()