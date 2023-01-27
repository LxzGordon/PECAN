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
    fig=plt.figure()
    color=['blue','red','green','orange','darkviolet']
    ax0,ax1=fig.subplots(1,2)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax1.set_xticks([])
    ax1.set_yticks([])
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

    for c in range(5):
        input_obs=np.reshape(obs[np.random.randint(600),:30],(30,5,4,20))
        #mep actions
        mep_actions=[]
        for _ in range(100):
            ind=np.random.randint(15)
            agent=partner[ind]
            prob=agent.direct_policy(input_obs)[:steps]
            #act=np.zeros((6))
            #ind=np.random.choice(np.arange(6),p=prob)
            #act[ind]=1
            #mep_actions.append(act)
            mep_actions.append(prob)
        
        mep_actions=np.vstack([act for act in mep_actions])
        print(mep_actions.shape)
        tsne1=TSNE(n_components=2)
        tsne1.fit_transform(mep_actions)
        mep_coor=np.array(tsne1.embedding_)

        #ens actions
        ens_actions=[]
        '''for _ in range(60):
            ind=np.random.randint(15)
            agent=partner[ind]
            prob=agent.direct_policy(input_obs)[:steps]
            #act=np.zeros((6))
            #ind=np.random.choice(np.arange(6),p=prob)
            #act[ind]=1
            #ens_actions.append(act)
            ens_actions.append(prob)'''

        for _ in range(100):
            group_ind=np.random.randint(3)
            if group_ind==0:
                ensemble=partner[0:5]
            elif group_ind==1:
                ensemble=partner[5:10]
            elif group_ind==2:
                ensemble=partner[10:15]
            
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
            ens_actions.append(probs)

        ens_actions=np.vstack([act for act in ens_actions])
        print(ens_actions.shape)
        tsne2=TSNE(n_components=2)
        tsne2.fit_transform(ens_actions)
        ens_coor=np.array(tsne2.embedding_)

        sns.scatterplot(mep_coor[:,0],mep_coor[:,1],color=color[c],ax=ax0)
        sns.scatterplot(ens_coor[:,0],ens_coor[:,1],color=color[c],ax=ax1)
    #seaborn.despine(top=False, right=False, left=False, bottom=False, offset=None, trim=False)
    plt.show()
tsne_actions()