import os
targets = ['crime', 'athlete', 'ceo', 'professor','celebrity']
for target in targets:
    cnt = 0
    path = './dataset/'+target+'/'
    for filename in os.listdir(path):
        f,ext = filename.split('.')
        if ext=='jpg':
            new_name = os.path.join(path,target+'_'+str(cnt)+'.'+ext)
            old_name = os.path.join(path,filename)
            os.rename(old_name,new_name)
            cnt+=1
    print(target+"done")