import random
import re

def main():

    #read file into mem
    f1 = open('jak2_data.csv')
    all_lines = f1.readlines()

    generate_num =50
    smileLen = 60
    #get a random sample, without repetitions
    lines = random.sample(all_lines[1:len(all_lines)], generate_num)
    lines.sort(key=len)
    count =0
    fileName = 'data_'+str(generate_num)+'_len'+str(smileLen)+'.smi'
    with open(fileName, 'w') as f:
        for data in lines:
            smile = data.split(",")[0]
            if len(smile) < smileLen :
              data = smile + " ," +  data.split(",")[1] 
              f.write("%s" % data) 
              count= count +1


    print( 'The data file is',fileName,'with size:',count)
    
    
#    ti = ["tiago","Oliveira","Pereirs"]
#    numb = ["1","2","3"]
#    results = []
#    
#    fileName = 'lixo.smi'
#    with open(fileName, 'w') as f:
#        for i,cl in enumerate(ti):
#          data = str(ti[i]) + "," +  str(numb[i])
#          results.append(data) 
#          f.write("%s\n" % data)
#      
#    print(results)
if __name__ == "__main__":
    main()
  
