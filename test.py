# GPT-4o Vision Test
print("\nTesting GPT-4o Vision...")
try:
    import openai
    client = openai.OpenAI(api_key="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")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5
    )
    print("✅ GPT-4o: WORKING")
except Exception as e:
    print(f"❌ GPT-4o ERROR: {e}")