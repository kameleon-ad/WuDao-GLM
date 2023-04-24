from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

from absl import app


GLM = "THUDM/glm-10b-chinese"
ChatGLM = "THUDM/chatglm-6b"


def test_chat_glm():
    tokenizer = AutoTokenizer.from_pretrained(ChatGLM, trust_remote_code=True)
    model = AutoModel.from_pretrained(ChatGLM, trust_remote_code=True).half().cuda()

    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    print(response)


def test_glm():
    tokenizer = AutoTokenizer.from_pretrained(GLM, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(GLM, trust_remote_code=True)
    model = model.half().cuda()

    inputs = tokenizer("凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。",
                       return_tensors="pt")
    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
    inputs = {key: value.cuda() for key, value in inputs.items()}
    outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
    print(tokenizer.decode(outputs[0].tolist()))


def main(_):
    print("[INFO] ChatGLM")
    test_chat_glm()
    print("[INFO] GLM")
    test_glm()
    return 0


if __name__ == "__main__":
    app.run(main)
