from google import genai
import os


class Generator:
    def __init__(self):
        print("Initializing Gemini client...")

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=api_key)

        print("Gemini client initialized!")

    def generate_answer(self, question, context):

        prompt = f"""
You are a personal RAG assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:
"I couldn't find that information in the provided documents."

Do not invent or assume information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return response.text


if __name__ == "__main__":

    generator = Generator()

    question = "Tell me about my LSTM anomaly detection project."

    context = """
    Satellite Telemetry Anomaly Detection | Python, TensorFlow,
    LSTM Autoencoder, Streamlit.

    Built an LSTM Autoencoder to detect anomalies in multivariate
    satellite telemetry, trained on NASA's SMAP satellite telemetry
    dataset to flag abnormal behavior indicative of potential faults.

    Engineered the full pipeline including data loading,
    preprocessing, model training, and anomaly scoring.
    """

    answer = generator.generate_answer(question, context)

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(answer)