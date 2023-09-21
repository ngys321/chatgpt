import json

def xlcost2cosqa(input_file_path, output_file_path):

    with open(input_file_path, "r") as f:
        lines = f.readlines()

    with open(output_file_path, "w") as f:
        new_data_list = []
        for line in lines:
            data = json.loads(line)
            new_data_list.append(data)    
        json.dump(new_data_list, f, indent=4)



if __name__ == "__main__":
    # input_file_path = "cosqa_da_.json"
    # output_file_path = "cosqa_da.json"
    # xlcost2cosqa(input_file_path, output_file_path)

    # input_file_path = "cosqa_da_spt_.json"
    # output_file_path = "cosqa_da_spt.json"
    # xlcost2cosqa(input_file_path, output_file_path)

    # input_file_path = "cosqa_da_sro_.json"
    # output_file_path = "cosqa_da_sro.json"
    # xlcost2cosqa(input_file_path, output_file_path)

    # input_file_path = "cosqa_da_srm_.json"
    # output_file_path = "cosqa_da_srm.json"
    # xlcost2cosqa(input_file_path, output_file_path)

    input_file_path = "cosqa_da_spt_sro_.json"
    output_file_path = "cosqa_da_spt_sro.json"
    xlcost2cosqa(input_file_path, output_file_path)
    input_file_path = "cosqa_da_spt_srm_.json"
    output_file_path = "cosqa_da_spt_srm.json"
    xlcost2cosqa(input_file_path, output_file_path)
    input_file_path = "cosqa_da_spt_sro_srm_.json"
    output_file_path = "cosqa_da_spt_sro_srm.json"
    xlcost2cosqa(input_file_path, output_file_path)
