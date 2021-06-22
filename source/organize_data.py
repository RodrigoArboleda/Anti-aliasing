import os
from shutil import copyfile
from pathlib import Path

def rec_create_folder(path):
    try:
        os.stat(path)
    except:
        parent = str(Path(path).parent)
        rec_create_folder(parent)
        os.mkdir(path)


def organize(folder_name, folder_dest):

    path_save_files = os.path.abspath(os.getcwd()) + os.sep + folder_dest
    path = os.path.abspath(os.getcwd()) + os.sep + folder_name
    print(path_save_files)

    #se ainda nao tem o 
    rec_create_folder(path_save_files)

    #iterando sobre todos os arquivos do direto e de seus subdiretorios
    for subdir, dirs, files in os.walk(path):
        
        #se o que estou iterando agr eh um arquivo
        for filename in files:
            #pegando seu path e verificando se eh uma imagem
            filepath =  subdir + os.sep + filename
            if filepath.endswith(".jpeg") or filepath.endswith(".jpeg") or filepath.endswith(".png"):

                #colocando o arquivo na nova pasta com o novo nome    
                filepath_vet = filepath.split('/')
                new_filename = filepath_vet[-2] + '_' + filepath_vet[-1]
                new_filepath = path_save_files + os.sep + new_filename
                print(new_filepath)
                copyfile(filepath, new_filepath)

if __name__ == "__main__":
    folder_name = input()
    folder_dest = input()
    organize(folder_name, folder_dest)