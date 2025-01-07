# Mixture-of-Agents in 50 lines of code
import asyncio
import os

from together import AsyncTogether, Together

from dotenv import load_dotenv


load_dotenv()
# Get the API key from the environment variable
api_key = os.environ.get("TOGETHER_API_KEY")


# Initialize Together clients with the API key

client = Together(api_key=api_key)

async_client = AsyncTogether(api_key=api_key)


user_prompt = "What are some fun things to do in Croydon?"

reference_models = [

    "meta-llama/Llama-Vision-Free",
    "meta-llama/Llama-Vision-Free",
]

aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.


Responses from models:"""



async def run_llm(model):

    """Run a single LLM call with a reference model."""

    response = await async_client.chat.completions.create(

        model=model,

        messages=[{"role": "user", "content": user_prompt}],

        temperature=0.7,

        max_tokens=512,

    )

    print(model)

    return response.choices[0].message.content



async def main():

    results = await asyncio.gather(*[run_llm(model) for model in reference_models])


 # Print intermediate results

    print("Intermediate outputs from reference models:")

    for model, result in zip(reference_models, results):

        print(f"\n{model}:\n{result}\n")
    

 # Run the final aggregation model   

    finalStream = client.chat.completions.create(

        model=aggregator_model,

        messages=[

            {"role": "system", "content": aggregator_system_prompt},

            {"role": "user", "content": ",".join(str(element) for element in results)},

        ],

        stream=True,

    )


    for chunk in finalStream:

        print(chunk.choices[0].delta.content or "", end="", flush=True)



asyncio.run(main())