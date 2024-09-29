from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import os

class QwenOCRReader():
  def __init__(self) -> None:
    # self.model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
    #                                                     trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-3B-Instruct", torch_dtype=torch.bfloat16, device_map={"": "mps"}
    # )
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": "mps"}, offload_buffers=True # Added
    )
    self.model = self.model.to("mps")
    # print(self.model.hf_device_map)
    # self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-3B-Instruct", trust_remote_code=True, device_map={"": "mps"})
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, device_map={"": "mps"})
    device = torch.device('mps' if torch.backends.mps.is_available else 'cpu')
    print(device)

  def reader(self, image, question):
    # image_path = os.path.join(os.getcwd(), 'src', 'images', 'C181a.jpeg')
    # image = Image.open(image_path)
    messages = [
      {
          "role": "user",
          "content": [
              {
                  "type": "image",
                  "image": image
              },
              # {"type": "text", "text": "Extract the meter reading from the image"},
              # {"type": "text", "text": "What is the answer for each of the questions in the image?"},
              {"type": "text", "text": question},
              # {"type": "text", "text": "What is the student's name and date of the test?"},
          ],
      }
    ]

    # Preparation for inference
    text = self.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = self.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # inputs = inputs.to("cuda")
    inputs = inputs.to("mps")

    # Inference: Generation of the output
    generated_ids = self.model.generate(**inputs, max_new_tokens=500)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = self.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # response_ids = generated_ids_trimmed[0]
    # response_text = self.tokenizer.batch_decode(response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    
    result = output_text[0].replace('\\n', '\n')
    print(result)
    return output_text
    # return response_text

# if __name__ == '__main__':
#   qwen = QwenOCRReader()
#   qwen.reader()