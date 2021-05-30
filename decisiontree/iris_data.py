import matplotlib.pyplot as plt 

def read_iris(filename):
    f=open(filename,'r')
    dataset=f.readlines()[:-1]
    for i,data in enumerate(dataset):
        data=data.strip().split(',')
        dataset[i]=[float(d) for d in data[:4]]+[data[4]]       
    return dataset

def draw_iris(dataset,sort=False):
    sampleNum=len(dataset)
    fig=plt.figure(1)
    for i in range(2):
        for j in range(2):
            x=list(range(sampleNum))   
            y=[data[i*2+j] for data in dataset]
            if sort:
                y=sorted(y)
            ax=plt.subplot(2,2,i*2+j+1)         
            plt.scatter(x,y,s=2)
    plt.show()


if __name__=='__main__':
    filename='iris.data'
    dataset=read_iris(filename)
    draw_iris(dataset)
    