from transformers import AutoTokenizer, AutoModel

from absl import app


GLM = "BAAI/glm-10b-chinese"
ChatGLM = "THUDM/chatglm-6b"


def main(_):
    tokenizer = AutoTokenizer.from_pretrained(GLM, trust_remote_code=True)
    model = AutoModel.from_pretrained(GLM, trust_remote_code=True).half().cuda()

    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    print(response)
    return 0


if __name__ == "__main__":
    app.run(main)
