import os
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

# ================================
# 引入 TPU 专属的“交通警察”模块
# ================================
import torch_xla.distributed.xla_multiprocessing as xmp


# 将原本的所有逻辑包裹在这个函数中
def _mp_fn(index):
    # 1. 基础配置
    model_id = "Qwen/Qwen3-8B"
    dataset_id = "yahma/alpaca-cleaned"
    output_dir = "./alpaca-lora-tpu-output"

    # 2. 数据准备与预处理
    def generate_prompt(data_point):
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset(dataset_id)

    def tokenize_function(examples):
        prompts = [generate_prompt({
            "instruction": instruction,
            "input": inp,
            "output": output
        }) for instruction, inp, output in zip(examples['instruction'], examples['input'], examples['output'])]

        tokenized_inputs = tokenizer(prompts, truncation=True, max_length=384, padding="max_length")
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset['train'].column_names)

    # 3. 加载模型与配置 LoRA
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    # 只有 0 号核心负责打印，避免控制台被 8 个进程刷屏
    if index == 0:
        model.print_trainable_parameters()

    # 4. 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,

        # 👇 核心防 OOM：单卡降回安全的 2
        per_device_train_batch_size=2,

        # 👇 核心提速：废弃梯度累加，不再让 TPU 编译巨型图！
        gradient_accumulation_steps=1,

        # 👇 核心防 Bug：彻底关掉检查点，不用任何补丁魔法
        gradient_checkpointing=False,

        optim="adamw_torch_xla",
        learning_rate=3e-4,
        num_train_epochs=3,
        warmup_steps=36,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        dataloader_drop_last=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    # 5. 启动训练
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        args=training_args,
        data_collator=data_collator,
    )

    model.config.use_cache = False
    trainer.train()

    # 6. 保存 LoRA 权重 (交给 Trainer 统一保存，它内部会处理多核防冲突)
    trainer.save_model(os.path.join(output_dir, "final_adapter"))
    if index == 0:
        print("✅ 训练完成，LoRA 权重已保存！")


if __name__ == "__main__":
    # 执行脚本时，从这里拉起 8 个 TPU 核心
    xmp.spawn(_mp_fn, args=(), nprocs=None)
