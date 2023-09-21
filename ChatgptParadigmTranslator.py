import openai
import time
import signal
import os
from dotenv import load_dotenv
# .env 파일 불러오기
load_dotenv()
# 환경 변수 가져오기
openai.api_key = os.getenv('OPENAI_API_KEY')


def handle_timeout(signum, frame):
    raise TimeoutError("Code execution timed out.")
import re

def extract_code(s):
    s = s.lstrip("#").lstrip(" ")
    s = s.lstrip("Here is the code rewritten in an imperative programming paradigm:")
    s = s.lstrip("Here is the code rewritten in a functional programming paradigm:")
    s = s.lstrip("Here is the code in an imperative programming paradigm:")
    s = s.lstrip("Here is the code in a functional programming paradigm:")
    s = s.lstrip("Here is the code in imperative programming paradigm:")
    s = s.lstrip("Here is the code in functional programming paradigm:")
    
    s = s.lstrip("Here is the code")
    s = s.lstrip(":")
    s = s.lstrip("\n")
    s = s.lstrip("```").rstrip("```")

    s = s.lstrip("python")
    s = s.lstrip("\n")

    return s

def ChatgptParadigmTranslator(code, src_p, tgt_p):

    '''
    code: code to be translated
    src_p: source programming paradigm
    tgt_p: target programming paradigm
    '''

    # instruct
    if src_p == None:        
        instructions = [
            {
                "role": "system",
                "content": "Provide answers in Python",
            },
            {
                "role": "user",
                "content": f"Code:\n{code}\n---\n\
                    Rewrite the code above in a {tgt_p} programming paradigm. \
                    No explanation. Don't say anything like,\"Here is the code\".",
            }
        ]
    else:
        instructions = [
            {
                "role": "system",
                "content": "Provide answers in Python",
            },
            {
                "role": "user",
                "content": f"Code:\n{code}\n---\n\
                    The code above was written in a {src_p} programming paradigm. \
                    Rewrite the code above in a {tgt_p} programming paradigm. \
                    No explanation. Don't say anything like,\"Here is the code\".",
            }
        ]
    
    # generate
    wait_t = 10
    inference_not_done = True
    while inference_not_done:
        try:
            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(60) 
                            
            # generate
            results = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = instructions)
            inference_not_done = False
            wait_t /= 2

            signal.alarm(0)
        
        except Exception as e:
            print(f"Waiting {wait_t} seconds")
            print(f"Error was : {e}")
            time.sleep(wait_t)
            wait_t *= 2
            if wait_t > 1280: # 상한선 1280 초
                wait_t = 1280

    # # print response
    # print(f"{results['choices'][0]['message']['role'].capitalize()}: ")
    # print(f"{results['choices'][0]['message']['content']}")
    # print("="*120)
    # print()
    
    # extract code
    text = results['choices'][0]['message']['content']

    # remove "Here is the code"
    text = extract_code(text)

    return text