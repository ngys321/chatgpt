from RuleBasedParadigmClassifier import RuleBasedParadigmClassifier
# paradigm_, score = RuleBasedParadigmClassifier(code1)

from ChatgptParadigmTranslator import ChatgptParadigmTranslator
# code = ChatgptParadigmTranslator(code, scr_p, tgt_p)

import json
import argparse
import logging
import random


logger = logging.getLogger(__name__)

def codeStr2codeToken(code_string):

    import io
    import tokenize
    import ast

    code_bytes = code_string.encode('utf-8')
    tokens = []

    for token in tokenize.tokenize(io.BytesIO(code_bytes).readline):
        if token.type == tokenize.NEWLINE:
            tokens.append('NEW_LINE')
        elif token.type == tokenize.INDENT:
            tokens.append('INDENT')
        elif token.type == tokenize.DEDENT:
            tokens.append('DEDENT')
        else:
            tokens.append(token.string)
     
        if 'utf-8' in tokens:# 'utf-8' 토큰 삭제
            tokens = [x for x in tokens if x != 'utf-8']
        if '\n' in tokens:# '\n' 토큰 삭제
            tokens = [x for x in tokens if x != '\n']
        if '' in tokens:# '' 토큰 삭제
            tokens = [x for x in tokens if x != '']

    return tokens

def codeToken2codeStr(tokens):

    code = ""
    indentation_level = 0

    for token in tokens:
        if token == "NEW_LINE":
            code += "\n"
            code += "    " * indentation_level
        elif token == "INDENT":
            indentation_level += 1
            code += "    "
        elif token == "DEDENT":
            indentation_level -= 1
            code = code[:-4]
        else:
            code += token + " "

    codeStr = code.strip()

    return codeStr

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

def remove_hereisthecode_sorry(input_file_path, output_file_path):

    # jsonl file. e.g. xlcost.jsonl
    if input_file_path.split(".")[-1] == "jsonl":
        with open(input_file_path, 'r') as f:
            lines = f.readlines()
        
        with open(output_file_path, 'w') as f:
            for line in lines:
                data = json.loads(line)
                
                codeStr = codeToken2codeStr(data['code_tokens'])
                if 'sorry' in codeStr:
                    continue
                elif 'Here is the code' in codeStr:
                    extracted_codeStr = extract_code(codeStr)
                    try:
                        data['code_tokens'] = codeStr2codeToken(extracted_codeStr) # 에러발생 위험
                    except:
                        data['code_tokens'] = extracted_codeStr.split()
                
                f.write(json.dumps(data) + "\n")


    # json file. e.g. cosqa.json
    elif input_file_path.split(".")[-1] == 'json':
        with open(input_file_path, "r") as f:
            lines = f.readlines()

        with open(output_file_path, "w") as f:
            for line in lines:
                data = json.loads(line)

                if 'sorry' in data['code_tokens']: # code_tokens에 sorry가 있으면 그 줄을 지움
                    continue
                elif 'Here is the code' in data['code_tokens']: # code_tokens에 Here is the code가 있으면, code 추출하기
                    data['code_tokens'] = extract_code(data['code_tokens'])
                
                f.write(json.dumps(data) + "\n")

def xlcost2cosqa(input_file_path, output_file_path):

    with open(input_file_path, "r") as f:
        lines = f.readlines()

    with open(output_file_path, "w") as f:
        new_data_list = []
        for line in lines:
            data = json.loads(line)
            new_data_list.append(data)    
        json.dump(new_data_list, f, indent=4)

def paradigm_flipper(input_paradigm):
    if input_paradigm == "imperative":
        return "functional"
    elif input_paradigm == "functional":
        return "imperative"
    else:
        raise ValueError("Invalid paradigm input")
    
def statement_random_ordering(code: str) -> str:
    """Shuffle the lines of a given code."""
    lines = code.strip().split("\n")
    random.shuffle(lines)
    return "\n".join(lines)

def statement_random_masking(code: str) -> str:
    lines =code.strip().split("\n")
    # lines 리스트가 비어있지 않을 경우에만 작업 수행
    if lines:
        random_index = random.randint(0, len(lines) - 1)
        lines[random_index] = "<masked statement>"
    return "\n".join(lines)

def chatgpt_paradigm_da(input_file_path, output_file_path, do_spt=False, do_sro=False, do_srm=False):

    '''
    input_file_path: input file path
    output_file_path: output file path
    do_spt: whether to use source paradigm tagging(spt) or not
    do_sro: whether to use statement random ordering(sro) or not
    do_srm: whether to use statement random masking(srm) or not
    '''

    # jsonl file. e.g. xlcost.jsonl
    if input_file_path.split(".")[-1] == "jsonl":

        # read
        with open(input_file_path, "r") as f:
            lines = f.readlines()

        # max index checking
        idx_list = []
        for line in lines:
            data = json.loads(line)
            idx = data["idx"] 
            idx_list.append(idx)
        max_idx = max(idx_list)

        # write
        with open(output_file_path, "w") as f:
            for line in lines:
                data = json.loads(line)

                # get code
                code = codeToken2codeStr(data["code_tokens"])
                # paradigm classification
                src_p, score = RuleBasedParadigmClassifier(code)

                # paradigm translation
                if do_spt and (src_p == "imperative" or src_p == "functional"): # src_p가 hybrid 인 경우는 제외
                    tgt_p = paradigm_flipper(src_p)
                    if do_spt:
                        code_w_spt = ChatgptParadigmTranslator(code, src_p, tgt_p)
                    else:
                        code_wo_spt = ChatgptParadigmTranslator(code, None, tgt_p)

                # paradigm translation with statement random ordering(sro)
                if do_sro and (src_p == "imperative" or src_p == "functional"): # src_p가 hybrid 인 경우는 제외
                    shuffled_code = statement_random_ordering(code)
                    tgt_p = paradigm_flipper(src_p)
                    if do_spt:
                        code_w_spt_w_sro = ChatgptParadigmTranslator(shuffled_code, src_p, tgt_p)
                    else:
                        code_wo_spt_w_sro = ChatgptParadigmTranslator(shuffled_code, None, tgt_p)

                
                # paradigm translation with statement random masking(srm)
                if do_srm and (src_p == "imperative" or src_p == "functional"): # src_p가 hybrid 인 경우는 제외
                    masked_code = statement_random_masking(code)
                    tgt_p = paradigm_flipper(src_p)
                    if do_spt:
                        code_w_spt_w_srm = ChatgptParadigmTranslator(masked_code, src_p, tgt_p)
                    else:
                        code_wo_spt_w_srm = ChatgptParadigmTranslator(masked_code, None, tgt_p)
                
                

                # add original data
                new_data = {}
                new_data["idx"] = data["idx"]
                new_data["docstring_tokens"] = data["docstring_tokens"]
                new_data["code_tokens"] = data["code_tokens"]
                new_data['url'] = str(data['idx']) + "-Python"
                f.write(json.dumps(new_data) + "\n")

                # add augmented data
                if do_spt and (src_p == "imperative" or src_p == "functional"): # src_p가 hybrid 인 경우는 제외
                    new_data = {}
                    new_data["idx"] = max_idx + data["idx"]
                    new_data["docstring_tokens"] = data["docstring_tokens"]
                    if do_spt:
                        # new_data["code_tokens"] = codeStr2codeToken(code_w_spt) # <- 에러발생 위험. 가끔 code_w_spt가 parse되지 않는 경우가 있음.
                        try:
                            new_data["code_tokens"] = codeStr2codeToken(code_w_spt)
                        except:
                            new_data["code_tokens"] = code_w_spt.split()
                                                                                # try except로 처리할까?
                                                                                # try:
                                                                                #    new_data["code_tokens"] = codeStr2codeToken(code_w_spt)
                                                                                # except:
                                                                                #    new_data["code_tokens"] = code_w_spt.split()
                    else:
                        # new_data["code_tokens"] = codeStr2codeToken(code_wo_spt) # <- 에러발생 위험. 가끔 code_wo_spt가 parse되지 않는 경우가 있음.
                        try:
                            new_data["code_tokens"] = codeStr2codeToken(code_wo_spt)
                        except:
                            new_data["code_tokens"] = code_wo_spt.split()
                                                                                # try except로 처리할까?
                                                                                # try:
                                                                                #    new_data["code_tokens"] = codeStr2codeToken(code_w_spt)
                                                                                # except:
                                                                                #    new_data["code_tokens"] = code_wo_spt.split()
                    new_data['url'] = str(max_idx + data["idx"]) + "-Python"
                    f.write(json.dumps(new_data) + "\n")

                # add augmented data with statement random ordering(sro)
                if do_sro and (src_p == "imperative" or src_p == "functional"): # src_p가 hybrid 인 경우는 제외
                    new_data = {}
                    new_data["idx"] = 2 * max_idx + data["idx"]
                    new_data["docstring_tokens"] = data["docstring_tokens"]
                    if do_spt:
                        # new_data["code_tokens"] = codeStr2codeToken(code_w_spt_w_sro) # <- 에러발생 위험. 가끔 code_w_spt_w_sro가 parse되지 않는 경우가 있음.
                        try:
                            new_data["code_tokens"] = codeStr2codeToken(code_w_spt_w_sro)
                        except:
                            new_data["code_tokens"] = code_w_spt_w_sro.split()
                                                                                # try except로 처리할까?
                                                                                # try:
                                                                                #    new_data["code_tokens"] = codeStr2codeToken(code_w_spt_w_sro)
                                                                                # except:
                                                                                #    new_data["code_tokens"] = code_w_spt_w_sro.split()
                    else:
                        # new_data["code_tokens"] = codeStr2codeToken(code_wo_spt_w_sro) # <- 에러발생 위험. 가끔 code_wo_spt_w_sro가 parse되지 않는 경우가 있음.
                        try:
                            new_data["code_tokens"] = codeStr2codeToken(code_wo_spt_w_sro)
                        except:
                            new_data["code_tokens"] = code_wo_spt_w_sro.split()
                                                                                # try except로 처리할까?
                                                                                # try:
                                                                                #    new_data["code_tokens"] = codeStr2codeToken(code_wo_spt_w_sro)
                                                                                # except:
                                                                                #    new_data["code_tokens"] = code_wo_spt_w_sro.split()
                    new_data['url'] = str(2 * max_idx + data["idx"]) + "-Python"
                    f.write(json.dumps(new_data) + "\n")

                
                # add augmented data with statement random masking(srm)
                if do_srm and (src_p == "imperative" or src_p == "functional"): # src_p가 hybrid 인 경우는 제외
                    new_data = {}
                    new_data['idx'] = 3 * max_idx + data['idx']
                    new_data['docstring_tokens'] = data['docstring_tokens']
                    if do_spt:
                        # new_data['code_tokens'] = codeStr2codeToken(code_w_spt_w_srm) # <- 에러발생 위험. 가끔 code_w_spt_w_srm가 parse되지 않는 경우가 있음.
                        try:
                            new_data['code_tokens'] = codeStr2codeToken(code_w_spt_w_srm)
                        except:
                            new_data['code_tokens'] = code_w_spt_w_srm.split()
                                                                                # try except로 처리할까?
                                                                                # try:
                                                                                #    new_data["code_tokens"] = codeStr2codeToken(code_w_spt_w_srm)
                                                                                # except:
                                                                                #    new_data["code_tokens"] = code_w_spt_w_srm.split()
                    else:
                        # new_data['code_tokens'] = codeStr2codeToken(code_wo_spt_w_srm) # <- 에러발생 위험. 가끔 code_wo_spt_w_srm가 parse되지 않는 경우가 있음.
                        try:
                            new_data['code_tokens'] = codeStr2codeToken(code_wo_spt_w_srm)
                        except:
                            new_data['code_tokens'] = code_wo_spt_w_srm.split()
                                                                                # try except로 처리할까?
                                                                                # try:
                                                                                #    new_data["code_tokens"] = codeStr2codeToken(code_wo_spt_w_srm)
                                                                                # except:
                                                                                #    new_data["code_tokens"] = code_wo_spt_w_srm.split()
                    new_data['url'] = str(3 * max_idx + data["idx"]) + "-Python"
                    f.write(json.dumps(new_data) + "\n")

                

    # json file. e.g. cosqa.json
    elif input_file_path.split(".")[-1] == "json":

        # read
        with open(input_file_path, "r") as f:
            data = json.load(f)

        # max index checking
        idx_list = []
        for d in data:
            idx = int(d["idx"].split("-")[-1])
            idx_list.append(idx)
        max_idx = max(idx_list)

        # write
        with open(output_file_path, "w") as f:
            for d in data:

                # get code
                code = d["code_tokens"] # cosqa. should i use code or code_tokens? use code but remove the comment in it. I made the code_tokens from code, removing the comment.
                # paradigm classification
                src_p, score = RuleBasedParadigmClassifier(code)

                # paradigm translation
                if do_spt and (src_p == "imperative" or src_p == "functional"):
                    tgt_p = paradigm_flipper(src_p)
                    if do_spt:
                        code_w_spt = ChatgptParadigmTranslator(code, src_p, tgt_p)
                    else:
                        code_wo_spt = ChatgptParadigmTranslator(code, None, tgt_p)
        
                # paradigm translation with statement random ordering(sro)
                if do_sro and (src_p == "imperative" or src_p == "functional"): # src_p가 hybrid 인 경우는 제외
                    shuffled_code = statement_random_ordering(code)
                    tgt_p = paradigm_flipper(src_p)
                    if do_spt:
                        code_w_spt_w_sro = ChatgptParadigmTranslator(shuffled_code, src_p, tgt_p)
                    else:
                        code_wo_spt_w_sro = ChatgptParadigmTranslator(shuffled_code, None, tgt_p)

                
                # paradigm translation with statement random masking(srm)
                if do_srm and (src_p == "imperative" or src_p == "functional"): # src_p가 hybrid 인 경우는 제외
                    masked_code = statement_random_masking(code)
                    tgt_p = paradigm_flipper(src_p)
                    if do_spt:
                        code_w_spt_w_srm = ChatgptParadigmTranslator(masked_code, src_p, tgt_p)
                    else:
                        code_wo_spt_w_srm = ChatgptParadigmTranslator(masked_code, None, tgt_p)
                

                # add original data
                new_data = {}
                new_data["idx"] = d["idx"]
                new_data['doc'] = d['doc']
                new_data["code_tokens"] = d["code_tokens"]
                new_data["docstring_tokens"] = d["docstring_tokens"]
                new_data["label"] = d["label"]
                new_data['retrieval_idx'] = int(d["idx"].split("-")[-1])
                f.write(json.dumps(new_data) + "\n")

                # add augmented data
                if do_spt and (src_p == "imperative" or src_p == "functional"):
                    new_data = {}
                    new_data["idx"] = "cosqa-train-"+str(int(max_idx) + int(d["idx"].split("-")[-1]))
                    new_data['doc'] = d['doc']
                    if do_spt:
                        new_data["code_tokens"] = code_w_spt
                    else:
                        new_data["code_tokens"] = code_wo_spt
                    new_data["docstring_tokens"] = d["docstring_tokens"]
                    new_data["label"] = d["label"]
                    new_data['retrieval_idx'] = int(max_idx) + int(d["idx"].split("-")[-1])
                    f.write(json.dumps(new_data) + "\n")

                # add augmented data with statement random ordering(sro)
                if do_sro and (src_p == "imperative" or src_p == "functional"): # src_p가 hybrid 인 경우는 제외
                    new_data = {}
                    new_data["idx"] = "cosqa-train-"+str(int(2 * max_idx) + int(d["idx"].split("-")[-1]))
                    new_data['doc'] = d['doc']
                    if do_spt:
                        new_data["code_tokens"] = code_w_spt_w_sro
                    else:
                        new_data["code_tokens"] = code_wo_spt_w_sro
                    new_data["docstring_tokens"] = d["docstring_tokens"]
                    new_data["label"] = d["label"]
                    new_data['retrieval_idx'] = int(2 * max_idx) + int(d["idx"].split("-")[-1])
                    f.write(json.dumps(new_data) + "\n")

                
                # add augmented data with statement random masking(srm)
                if do_srm and (src_p == "imperative" or src_p == "functional"): # src_p가 hybrid 인 경우는 제외
                    new_data = {}
                    new_data["idx"] = "cosqa-train-"+str(int(3 * max_idx) + int(d["idx"].split("-")[-1]))
                    new_data['doc'] = d['doc']
                    if do_spt:
                        new_data["code_tokens"] = code_w_spt_w_srm
                    else:
                        new_data["code_tokens"] = code_wo_spt_w_srm
                    new_data["docstring_tokens"] = d["docstring_tokens"]
                    new_data["label"] = d["label"]
                    new_data['retrieval_idx'] = int(3 * max_idx) + int(d["idx"].split("-")[-1])
                    f.write(json.dumps(new_data) + "\n")
                

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--do_spt", action="store_true", help="whether to use source paradigm tagging(spt) or not")

    parser.add_argument("--do_sro", action="store_true", help="whether to use statement random ordering(sro) or not")

    parser.add_argument("--do_srm", action="store_true", help="whether to use statement random masking(srm) or not")

    args = parser.parse_args()

    logger.info(args)



    # # --do_spt 했을때
    # if args.do_spt and not args.do_sro:
    #     # data augmentation
    #     in_cosqa = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa.json'
    #     out_cosqa = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa_da_spt_.json'
    #     chatgpt_paradigm_da(in_cosqa, out_cosqa, do_spt=args.do_spt )
    #     in_xlcost = '/home/ysnamgoong42/ws/chatgpt/dataset/xlcost.jsonl'
    #     out_xlcost = '/home/ysnamgoong42/ws/chatgpt/dataset/xlcost_da_spt.jsonl'
    #     chatgpt_paradigm_da(in_xlcost, out_xlcost, do_spt=args.do_spt )
    #     # remove_hereisthecode_sorry
    #     input_file_path = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa_da_spt_.json'
    #     output_file_path = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa_da_spt_.json'
    #     remove_hereisthecode_sorry(input_file_path, output_file_path)
    #     input_file_path = '/home/ysnamgoong42/ws/chatgpt/dataset/xlcost_da_spt.jsonl'
    #     output_file_path = '/home/ysnamgoong42/ws/chatgpt/dataset/xlcost_da_spt.jsonl'
    #     remove_hereisthecode_sorry(input_file_path, output_file_path)
    #     # xlcost2cosqa: format 변경. cosqa 만 해줌. xlcost는 안해도됨.
    #     input_file_path = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa_da_spt_.json'
    #     output_file_path = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa_da_spt.json'
    #     xlcost2cosqa(input_file_path, output_file_path)


    # # --do_spt --do_sro 했을때
    # if args.do_spt and args.do_sro:
    #     # data augmentation
    #     in_cosqa = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa.json'
    #     out_cosqa = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa_da_spt_sro_.json'
    #     chatgpt_paradigm_da(in_cosqa, out_cosqa, do_spt=args.do_spt, do_sro=args.do_sro, do_srm=args.do_srm )
    #     in_xlcost = '/home/ysnamgoong42/ws/chatgpt/dataset/xlcost.jsonl'
    #     out_xlcost = '/home/ysnamgoong42/ws/chatgpt/dataset/xlcost_da_spt_sro.jsonl'
    #     chatgpt_paradigm_da(in_xlcost, out_xlcost, do_spt=args.do_spt, do_sro=args.do_sro, do_srm=args.do_srm )
    #     # remove_hereisthecode_sorry
    #     input_file_path = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa_da_spt_sro_.json'
    #     output_file_path = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa_da_spt_sro_.json'
    #     remove_hereisthecode_sorry(input_file_path, output_file_path)
    #     input_file_path = '/home/ysnamgoong42/ws/chatgpt/dataset/xlcost_da_spt_sro.jsonl'
    #     output_file_path = '/home/ysnamgoong42/ws/chatgpt/dataset/xlcost_da_spt_sro.jsonl'
    #     remove_hereisthecode_sorry(input_file_path, output_file_path)
    #     # xlcost2cosqa: format 변경. cosqa 만 해줌. xlcost는 안해줌.
    #     input_file_path = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa_da_spt_sro_.json'
    #     output_file_path = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa_da_spt_sro.json'
    #     xlcost2cosqa(input_file_path, output_file_path)



    # Process the cosqa dataset
    in_cosqa = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa.json'
    out_cosqa = '/home/ysnamgoong42/ws/chatgpt/dataset/cosqa_DAOUTPUT.json' # <- DAOUTPUT 부분만 이름 고쳐서 쓰면 됨.
    chatgpt_paradigm_da(in_cosqa, out_cosqa, do_spt=args.do_spt, do_sro=args.do_sro, do_srm=args.do_srm )
    remove_hereisthecode_sorry(out_cosqa, out_cosqa) # remove_hereisthecode_sorry
    xlcost2cosqa(out_cosqa, out_cosqa) # xlcost2cosqa: format 변경. cosqa 만 해줌. xlcost는 안해줌.

    # Process the xlcost dataset
    in_xlcost = '/home/ysnamgoong42/ws/chatgpt/dataset/xlcost.jsonl'
    out_xlcost = '/home/ysnamgoong42/ws/chatgpt/dataset/xlcost_DAOUTPUT.jsonl' # <- DAOUTPUT 부분만 이름 고쳐서 쓰면 됨.
    chatgpt_paradigm_da(in_xlcost, out_xlcost, do_spt=args.do_spt, do_sro=args.do_sro, do_srm=args.do_srm )
    remove_hereisthecode_sorry(out_xlcost, out_xlcost) # remove_hereisthecode_sorry


# <command examples>

# python chatgpt_paradigm_da.py --do_spt

# python chatgpt_paradigm_da.py --do_sro

# python chatgpt_paradigm_da.py --do_srm

# python chatgpt_paradigm_da.py --do_spt --do_sro

# python chatgpt_paradigm_da.py --do_spt --do_srm

# python chatgpt_paradigm_da.py --do_sro --do_srm

# python chatgpt_paradigm_da.py --do_spt --do_sro --do_srm

