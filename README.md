# chatgpt

```
참고:

**spt** (source paradigm tagging) - 챗지피티 프롬프트에 코드 넣을때, 그 코드의 패러다임이 뭔지 알려주는것

**sro** (statement random ordering) - 챗지피티 프롬프트에 코드 넣을때, 원본 코드의 라인들을 무작위로 섞어서 넣어주는 것. 좀더 다양한 코드를 생성하기 위함.

**srm** (statement random masking) - 챗지피티 프롬프트에 코드 넣을때, 원본 코드의 라인들 중 무작위로 하나를 <masked statement> 로 바꾸는 것. 좀더 다양한 코드를 생성하기 위함.
```

**chatgpt**

dataset

**xlcost.jsonl                                {원본 } 데이터**

**xlcost_da_spt.jsonl                   {원본 + (spt)aug }  데이터**

xlcost_da_sro.jsonl                      {sro aug } 데이터

~~xlcost_da_srm.jsonl                     {srm aug (생성중) } 데이터~~

**xlcost_da_spt_sro.jsonl           {원본 + (spt)aug + sro aug } 데이터**

**~~xlcost_da_spt_srm.jsonl          {원본 + (spt)aug + srm aug } 데이터~~**

—

**cosqa.json                                  {원본} 데이터**

**cosqa_da_spt.json                    {원본 + (spt)aug } 데이터**

cosqa_da_sro.json                      {sro aug } 데이터

~~cosqa_da_srm.json                     {srm aug (생성중) } 데이터~~

**cosqa_da_spt_sro.json            {원본 + (spt)aug + sro aug } 데이터**

**~~cosqa_da_spt_srm.json           {원본 + (spt)aug + srm aug } 데이터~~**

—

- **chatgpt_paradigm_da.py**        코드 패러다임 변환해서 저장하는 코드
    
    
                                                 **<if __name__ == “__main__”>**
    
                                                       **필요에 맞게 함수 사용.**
    
                                                       e.g.
    
                                                       # data aug
                                                        in_cosqa = …
                                                        out_cosqa = …
                                                        chatgpt_paradigm_da ( in_cosqa, out_cosqa, …)
    
                                                        # remove_hereisthecode_sorry
                                                        remove_hereisthecode_sorry ( out_cosqa, out_cosqa )
                                                        # xlcost format → cosqa format
                                                        out2_cosqa = …
                                                        xlcost2cosqa ( out_cosqa, out2_cosqa )
                                                       
    
                                                      **<핵심 함수>**
    
                                                      **chatgpt_paradigm_da(**
    
                                                              **input_file_path,**
    
                                                              **output_file_path,**
    
                                                              **do_spt=False,**
    
                                                              **do_sro=False,**
    
                                                              **do_srm=False)**
    
                                                              
    
                                                    - input 데이터 파일 읽어 들임.
    
                                                    - max index 값 저장.
    
                                                    - 하나씩 iteration 돌면서,
    
                                                      **코드 패러다임 변환** (do_sro 나 do_srm 했으면, 
    
                                                      추가로 패러다임 변환 더 함.)
    
                                                    - 코드 저장 ( 원본코드 + 변환코드 + do_sro 했으면,
    
                                                      추가 변환 코드 + do_srm 했으면 추가 변환 코드)
    
                                                    ⇒ 위 과정이 
    
                                                         if xlcost 인 경우와 
    
                                                         elif cosqa 인 경우로 나눠져 있음
    
                                                      <**그 외 함수들 설명>**
    
                                                      **codeStr2codeToken(str) → list**
    
                                                          코드 string 을 코드 토큰 list 로 변환
    
                                                      **codeToken2codeStr(list) → str**
    
                                                          코드 토큰 list 를 코드 string 으로 변환
    
                                                      **extract_code(str) → str**
    
                                                          코드 string 에서 앞부분에 Here is the code 하는
    
                                                          부분 잘라내고, 코드 부분만 추출
    
                                                      **remove_hereisthecode_sorry(input_file_path, output_file_path)**
    
                                                          코드 string 에 sorry 라는 문자열이 있는 경우는,
    
                                                          그 데이터 샘플 제외 시킴
    
                                                          코드 string 에 Here is the code 라는 문자열이 
    
                                                          있는 경우는, extract_code() 함수써서 
    
                                                          코드 부분만 추출
    
                                                          결과물은 output_file 에 저장.
    
                                               **xlcost2cosqa(input_file_path, output_file_path)**
    
                                                     xlcost 포맷의 데이터 파일 
    
                                                    → cosqa 포맷의 데이터 파일로 변환
    
                                                    **paradigm_flipper(str)→str**
    
                                                          “imperative” 들어오면, “functional” 리턴
    
                                                          “functional” 들어오면, “imperative” 리턴
    
                                              **statement_random_ordering(str) →str**
    
                                                         코드 string 들어오면, statement 순서 램덤으로 섞어,
    
                                                         그 결과 string 을 리턴
    
                                              **statement_random_masking(str) →str**
    
                                                         코드 string 들어오면, statement 중에 랜덤으로
    
                                                         하나를 <masked statement> 로 바꿈
    
                                                         그 결과 string 을 리턴
    

**ChatgptParadigmTranslator.py**     코드 패러다임 변환기 함수 정의. openai.api_key 값 익명화!

```
**<프롬프트>**

Code:
**{코드 내용} <- 코드 다양성 증진을 위해서,
             위에서 말한 statement_random_ordering() 함수나
             statement_random_masking() 함수로 코드 string 을 변형시킨 후에,
             넣어줌.**
---
Rewrite the code above in a {target 패러다임 이름} programming paradigm.
No explanation. Don't say anything like, "Here is the code".   
```

- **RuleBasedParadigmClassifier.py**  코드 패러다임 분류기 함수 정의
    
                  line 3 - line 627: 코드 string 에서 feature 뽑는 함수들 엄청 많음. 이해할 필요 없음
    
                  codeToken2codeStr() - 안중요
    
                  class ParadigmFeature() - 어떤 feature 를 가졌는지 저장하기 위한 class 
    
                  FeatureChecker() - 맨위에서 정의된 feature 뽑는 함수들 사용해서, 코드 string 이 어떤 feature 를 가졌는지 판단하고, 그 판단정보를 리턴
    
                **<핵심 함수>**
    
                **RuleBasedParadigmClassifier()**
    
                  - 코드 string 에서 feature 를 뽑는다.
    
            - feature 중에 **함수형 feature** 가 하나라도 있으면,
    
              함수형 패러다임 코드로 분류한다.
    
            - **함수형 feature** 가 하나도 없으면서 **명령형 feature**가 
    
              있으면, 명령형 패러다임 코드로 분류한다.
    
            - **명령형 feature, 함수형 feature** 모두 없으면,
    
               Hybrid 패러다임 코드로 분류한다.