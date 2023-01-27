
import numpy as np
import torch as th
from human_aware_rl.baselines_utils import get_pbt_agent_from_config_eval

#from human_aware_rl.pbt.pbt import PBT_DATA_DIR

PBT_DATA_DIR='pbt_data_dir_2/'
# Visualization


def tsne_actions():

    # best_bc_models = load_pickle("data/bc_runs/best_bc_models")
    #seeds = [8015, 3554,  581, 5608, 4221]
    seed=9015
    from matplotlib import pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    sns.set(style='whitegrid')
    
    color=['blue','red','green']
    #ax0,ax1=fig.subplots(1,2)
    #ax0.set_xticks([])
    #ax0.set_yticks([])
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    items=8

    steps=2
    partner=[]
    for i in range(5):
        pbt_save_dir = './pbt_data_dir/pbt_simple/'
        partner.append(get_pbt_agent_from_config_eval(pbt_save_dir, 1, seed=seed, agent_idx=i,iter=1))
        partner.append(get_pbt_agent_from_config_eval(pbt_save_dir, 1, seed=seed, agent_idx=i,iter=152))
        partner.append(get_pbt_agent_from_config_eval(pbt_save_dir, 1, seed=seed, agent_idx=i,iter=305))

    print('Reading obs..')
    f=open('pbt_data_dir_2/unident_s/test1.txt','r')
    txt=f.readlines()
    obses=[]
    from tqdm import tqdm
    for t in txt:
        t=t[1:-2]
        t=t.split(',')
        obs=list(map(float,t))
        obs=th.Tensor(obs).cuda().view(1,400,5*4*20) #after dataloader, reshape to (-1,400,5,4,20)
        obses.append(obs)
    obs=th.cat([obs for obs in obses],dim=0)[:600].cpu().numpy()

    for iter in range(10):
        fig=plt.figure(figsize=(39,9))
        axes=[fig.add_subplot(2,items,i) for i in range(1,2*items+1)]

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_yticks([])
        for c in range(items):
            input_obs=np.reshape(obs[np.random.randint(600),:30],(30,5,4,20))
            #no grouping
            actions=[]
            for _ in range(500):
                ensemble=partner
                p=np.random.uniform(size=14)
                sorted_p=np.sort(p)/p.sum()
                w=np.zeros(shape=(15))
                w[0]=sorted_p[0]
                for i in range(1,14):
                    w[i]=sorted_p[i]-sorted_p[i-1]
                w[14]=1-sorted_p[13]

                w=np.reshape(w,(1,-1,1))
                w=np.repeat(w,steps,0)
                probs=[agent.direct_policy(input_obs)[:steps] for agent in ensemble]
                probs=np.concatenate([np.reshape(prob,(2,1,6)) for prob in probs],1)
                probs=(probs*w).sum(1)
                actions.append(probs)
            
            #actions=np.vstack([act for act in actions])
            #print(actions.shape)
            #tsne1=TSNE(n_components=2)
            #tsne1.fit_transform(actions)
            #coor=np.array(tsne1.embedding_)

            #ens actions
            #ens_actions=[]
            '''for _ in range(60):
                ind=np.random.randint(15)
                agent=partner[ind]
                prob=agent.direct_policy(input_obs)[:steps]
                #act=np.zeros((6))
                #ind=np.random.choice(np.arange(6),p=prob)
                #act[ind]=1
                #ens_actions.append(act)
                ens_actions.append(prob)'''

            color_ind=[]
            for _ in range(500):
                group_ind=np.random.randint(3)
                if group_ind==0:
                    ensemble=partner[0:5]
                elif group_ind==1:
                    ensemble=partner[5:10]
                elif group_ind==2:
                    ensemble=partner[10:15]
                color_ind.append(group_ind)
                color_ind.append(group_ind)

                p=np.random.uniform(size=4)
                sorted_p=np.sort(p)/p.sum()
                w=np.zeros(shape=(5))
                w[0]=sorted_p[0]
                for i in range(1,4):
                    w[i]=sorted_p[i]-sorted_p[i-1]
                w[4]=1-sorted_p[3]

                w=np.reshape(w,(1,-1,1))
                w=np.repeat(w,steps,0)
                probs=[agent.direct_policy(input_obs)[:steps] for agent in ensemble]
                probs=np.concatenate([np.reshape(prob,(2,1,6)) for prob in probs],1)
                probs=(probs*w).sum(1)

                #act=np.zeros((6))
                #ind=np.random.choice(np.arange(6),p=probs)
                #act[ind]=1
                #ens_actions.append(act)
                actions.append(probs)

            actions=np.vstack([act for act in actions])
            print(actions.shape)
            tsne2=TSNE(n_components=2)
            tsne2.fit_transform(actions)
            coor=np.array(tsne2.embedding_)
            sns.scatterplot(coor[:1000,0],coor[:1000,1],color='b',ax=axes[c])
            sns.scatterplot(coor[1000:,0],coor[1000:,1],color='r',ax=axes[c])
 

            for i,p in enumerate(coor[1000:]):
                p=np.reshape(p,(-1,2))
                sns.scatterplot(p[:,0],p[:,1],color=color[color_ind[i]],ax=axes[c+items])
        #seaborn.despine(top=False, right=False, left=False, bottom=False, offset=None, trim=False)
        #plt.show()

        plt.savefig('/home/lxz/Desktop/iter'+str(iter)+'.png')
tsne_actions()