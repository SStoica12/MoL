import os
import time


class OpenAICompatError(RuntimeError):
    pass


def create_client(api_key=None, base_url=None):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

    if not api_key:
        raise OpenAICompatError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return ("v1", OpenAI(**kwargs))
    except Exception:
        try:
            import openai
        except Exception as exc:
            raise OpenAICompatError(f"Failed to import openai package: {exc}") from exc

        openai.api_key = api_key
        if base_url:
            openai.api_base = base_url
        return ("legacy", openai)


def chat_completion(client_tuple, model, messages, temperature=0.0, max_tokens=16):
    client_kind, client = client_tuple
    if client_kind == "v1":
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    response = client.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"]


def chat_completion_with_retry(
    client_tuple,
    model,
    messages,
    temperature=0.0,
    max_tokens=16,
    max_retries=6,
    sleep_seconds=1.0,
):
    last_error = None
    for _ in range(max_retries):
        try:
            return chat_completion(
                client_tuple,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            last_error = exc
            time.sleep(sleep_seconds)

    raise OpenAICompatError(str(last_error) if last_error is not None else "OpenAI request failed.")
