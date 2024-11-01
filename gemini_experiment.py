import openai
import requests
import pandas as pd


# Load the dataset
def load_harmfulqa_dataset(file_path):
    return pd.read_csv(file_path)


# Function to query GPT (OpenAI API)
def query_gpt(prompt, api_key, model="gpt-4"):
    openai.api_key = api_key
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)


# Function to query Google Gemini (pseudo code as Gemini API may vary)
def query_google_gemini(prompt, api_key):
    url = "https://gemini.googleapis.com/v1/generate"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "maxTokens": 150
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()
        return response_data["responses"][0]["text"].strip()
    except Exception as e:
        return str(e)


# Evaluate prompt injections across LLMs
def evaluate_prompt_injection(dataset, gpt_api_key, gemini_api_key):
    results = []
    for index, row in dataset.iterrows():
        prompt = row['prompt']  # Assuming the dataset has a column named 'prompt'
        expected_behavior = row['expected_behavior']  # Assuming a column with expected behavior

        # Query GPT
        gpt_response = query_gpt(prompt, gpt_api_key)

        # Query Google Gemini
        gemini_response = query_google_gemini(prompt, gemini_api_key)

        # Record the responses
        result = {
            "Prompt": prompt,
            "Expected Behavior": expected_behavior,
            "GPT Response": gpt_response,
            "Gemini Response": gemini_response
        }
        results.append(result)
    return pd.DataFrame(results)


# Main function to execute the workflow
def main():
    # Set the API keys for OpenAI and Google Gemini
    gpt_api_key = "your_openai_api_key"
    gemini_api_key = "your_google_gemini_api_key"

    # Load harmfulQA dataset
    dataset_path = "harmfulQA_dataset.csv"
    harmfulqa_dataset = load_harmfulqa_dataset(dataset_path)

    # Evaluate prompt injections
    results_df = evaluate_prompt_injection(harmfulqa_dataset, gpt_api_key, gemini_api_key)

    # Save results to a CSV file
    results_df.to_csv("prompt_injection_results.csv", index=False)
    print("Evaluation completed. Results saved to 'prompt_injection_results.csv'")


if __name__ == "__main__":
    main()
