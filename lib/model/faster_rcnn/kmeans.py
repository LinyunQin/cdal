import torch


def kmeans(data, k=7, max_time=100):
    n, m = data.shape
    ini = torch.randint(n, (k,)).type(torch.long) #只有一维需要逗号
    midpoint = data[ini]   #随机选择k个起始点
    time = 0
    last_label = 0
    while(time < max_time):
        d = data.unsqueeze(0).repeat(k, 1, 1)   #shape k*n*m
        mid_ = midpoint.unsqueeze(1).repeat(1,n,1) #shape k*n*m
        dis = torch.sum((d - mid_)**2, 2)     #计算距离
        sse = torch.sum(torch.min(dis,0)[0])/len(data)
        label = dis.argmin(0)      #依据最近距离标记label
        if torch.sum(label != last_label)==0:  #label没有变化,跳出循环
            return sse
        last_label = label
        for i in range(k):  #更新类别中心点，作为下轮迭代起始
            kpoint = data[label==i]
            if kpoint.shape[0]==0:
                if i==0:
                    midpoint_next = midpoint[0].unsqueeze(0)
                else:
                    midpoint_next=torch.cat([midpoint_next, midpoint[i].unsqueeze(0)], 0)
            else:
                if i == 0:
                    midpoint_next = kpoint.mean(0).unsqueeze(0)
                else:
                    midpoint_next = torch.cat([midpoint_next, kpoint.mean(0).unsqueeze(0)], 0)
        midpoint = midpoint_next
        time += 1
    return sse