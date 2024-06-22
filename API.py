from fastapi import FastAPI , File , UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
import io
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import uvicorn
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

torch.cuda.empty_cache()

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
            "automatic-speech-recognition",
                model=model,
                    tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor,
                            max_new_tokens=128,
                                chunk_length_s=30,
                                    batch_size=16,
                                        return_timestamps=True,
                                            torch_dtype=torch_dtype,
                                                device=device,
                                                )


def analysis(content):
            model_id = "scb10x/llama-3-typhoon-v1.5x-8b-instruct"
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            messages=[
                    {
                        "role": "system",
                        "content": '''
                        # TASK TYPE #
                        You are expert in swindler call-center detection that classification Swindler or not
                        from text pattern
            
                        # INSTRUCTIONS#
                        Your objective is to classify and analyze Thai text from calling to detect
                        and prevent people from swindler
                        Example text from swindler
                        - สวัสดีครับ ผมเป็นเจ้าหน้าที่จาก DHL ไม่ทราบว่ากำลังเรียนคุณ (Name)ใช่ไหม ทางเราจะเรียนแจ้งว่าคุณมีพัสดุผิดกฎหมายตีคืนจาก (Some Place) หากคุณไม่มาชี้แจงของชิ้นนี้
                        เราจำเป็นต้องดำเนินคดีตามกฎหมายกับคุณ
                        - สวัสดีครับ ติดต่อจากสถานีตำรวจ...จากการตรวจสอบ พบว่าคุณตกเป็นผู้ต้องหาคดีฟอกเงินระดับประเทศเป็นคดีอาญามีชื่อคุณเป็นหนึ่งในผู้ต้องหาที่ต้องดำเนินคดี
                        หากคุณไม่ได้ทำและอยากยืนยันความบริสุทธิ์ คุณต้องให้ความร่วมมือกับตำรวจในการให้ข้อมูล
                        - เบอร์โทรศัพท์ภายใต้ชื่อของคุณมีผู้ร้องเรียนเข้ามาจำนวนมากจึงจะตัดสัญญาณโทรศัพท์ภายใน 2 ชั่วโมง
                        อยากรู้รายละเอียดให้กด 9 ติดต่อ กสทช.
                        - จากการตรวจสอบของผิดกฎหมายที่ทางตำรวจอายัดได้ พบว่าชื่อของคุณมีส่วนเกี่ยวข้องกับการกระทำผิดครั้งนี้
                        จึงขอตรวจสอบความเคลื่อนไหวของโทรศัพท์ของคุณว่าข้อมูลส่วนตัวรั่วไหลไปได้ยังไง อยากขอให้คุณติดตั้งแอปพลิเคชันที่เราส่งให้แล้วเดี๋ยวผมสอนวิธีให้
                        - สวัสดีค่ะ เนื่องจากคุณเป็นผู้ใช้ที่มีประวัติที่ดี ยินดีด้วยคุณถูก TikTok เลือกให้ทำงานหารายได้เสริมทางออนไลน์
                        ทำง่ายๆ จ่ายสดทุกบิล รายได้ต่อวัน 3,000บาท/วัน ติดต่อเรา @LINE(ID Line)
                        - สวัสดีครับ สถานีตำรวจภูธรเมืองเชียงใหม่ครับ ผม ร.ต.ท. (Name) ร้อยผู้บันทึก รหัส (ID) ขอทราบชื่อ เลขบัตรประจำตัว วันเดือนปีเกิด
                        ที่อยู่ตามบัตรประชาชนของผู้แจ้งความครับ ช่วยเล่าเหตุการณ์อย่างละเอียดด้วยครับว่าอะไรขึ้นถึงแจ้งความกับทางสถานีตำรวจ จากที่ฟังคุณเล่ามา
                        อาจเป็นเพราะข้อมูลของคุณรั่วไหล เพราะการจัดส่งของออกนอกประเทศจะต้องใช้บัตรประชาชน หรือเอกสารส่วนบุคคลเพื่อยืนยันในการจัดส่ง
                        เพื่อระบุว่าที่อยู่นี้ถูกจัดส่งถูกต้องตามขั้นตอนและมีที่มาที่ไปชัดเจน
                        - สวัสดีค่ะ คุณมีสิทธิ์กู้เงินสูงสุด 300,000 บาทพร้อมดอกเบี้ยสุดพิเศษ 3-7%..
            
                        # DO #
                        - analyze the text and warning if you confident that are swindler answer with นี่คือแก๊งค์คอลเซ็นเตอร์ โปรดหยุดการติดต่อหรือตัดสาย!!!
                        - warning in thai language
                        - if it is not swindler answer สายนี้ปลอดภัย
            
                        # DON'T #
                        - if you aren't confident don't warning and judge
                        - Translate Thai to English '''
                        ,
                    },
                            {
                        "role": "user",
                        "content": '''ผมชื่อ อาคม ขอโทษที่ต้องรบกวนเวลานะครับ เราตรวจพบว่าคุณมีส่วนพัวพันกับคดีฟอกเงินและสิ่งของผิดกฎหมายครับ 
                        ตอนนี้คุณต้องรีบดำเนินการแก้ไขครับ กรุณาแอดไลน์ของเจ้าหน้าที่ตำรวจในพื้นที่เพื่อพูดคุยรายละเอียดเพิ่มเติมครับ เมื่อคุณแอดไลน์แล้ว คุณจะเห็นชื่อว่า “สุรยุทธ” นายตำรวจนะครับ คุณสามารถติดต่อเขาได้เลยครับ สวัสดีครับ ผมสุรยุทธ เจ้าหน้าที่ตำรวจในพื้นที่จังหวัดกาญจนบุรีนะครับ ขอโทษที่ต้องรบกวนเวลานะครับ เรื่องที่คุณมีส่วนพัวพันกับคดีฟอกเงินและสิ่งของผิดกฎหมาย เราต้องการให้คุณโยกเงินในบัญชีทั้งหมดมาเพื่อตรวจสอบ หากไม่ทำตาม 
                        บัญชีธนาคารของคุณจะถูกอายัดและคุณจะมีความผิดครับ กรุณาโอนเงินมายังบัญชีที่ผมจะแจ้งให้ครับ''',
                    
                    }] # add message here
            
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.1,
                top_p=0.95,
            )
            response = outputs[0][input_ids.shape[-1]:]
            return tokenizer.decode(response, skip_special_tokens=True)



@app.post("/Voice-to-text")
async def voice_to_text(file: UploadFile = File(...)):
    voice_data = await file.read()

    result = pipe(voice_data)
    result = analysis(result['text'])
    return JSONResponse(content={"TEXT": result})

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0' ,port = 8000)
