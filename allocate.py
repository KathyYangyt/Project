import numpy as np
from torchvision import datasets, transforms

# the dataset allocation to clients when the data type is IID
def Cmnist_iid(dataset, num_clients):

    # print the data type
    print('type:iid')
    # the number images in each group
    num_group = int(len(dataset)/num_clients)
    # define the clients dictionary and index list
    dict_groups = {}
    idx_fragments_list = []
    # the number of index is the length of dataset
    for i in range(len(dataset)):
      idx_fragments_list.append(i)
    # allocate the dataset averagely
    for i in range(num_clients):

        dict_groups[i] = set(np.random.choice(idx_fragments_list, num_group, replace=False))
        idx_fragments_list = list(set(idx_fragments_list) - dict_groups[i]) 

    # print the allocation information
    print("The allocation of dataset")
    for i in range (num_clients):               
       print(len(dict_groups[i]),end=",")
       if((i+1)%10==0):
        print() 
    return dict_groups

# the equal dataset allocation to clients when the data type is non-IID
def Cmnist_noniid(dataset, num_clients):
    
    print('type:non_iid_euqal')
    # 12000 images: 200(num_fragments) × 60(group_imgs), each of user: 2×group_imgs
    num_fragments, group_imgs = 200, 60

    # idx_fragments_list: the index list of fragments
    idx_fragments_list = [i for i in range(num_fragments)]
    dict_groups = {i: np.array([]) for i in range(num_clients)}
    
    # get the label list and add the index to it
    indexs_labels = np.arange(len(dataset))
    labels_list=[]
    for i in range(len(dataset)):
      labels_list.append(dataset[i][1])

    # put the labels and their indexs to the stack and sort the labels
    idxs_labels_np_np = np.vstack((indexs_labels , labels_list))
    idxs_labels_np_np= idxs_labels_np_np[:,idxs_labels_np_np[1,:].argsort()]
    indexs_labels = idxs_labels_np_np[0,:]

    # Splitting the dataset and assigning it to clients, with 2 random fragments per user
    for i in range(num_clients):
        random_set = set(np.random.choice(idx_fragments_list, 2, replace=False))
        idx_fragments_list = list(set(idx_fragments_list) - random_set)
        for s in random_set:
            dict_groups[i] = np.concatenate((dict_groups[i], indexs_labels[s*group_imgs:(s+1)*group_imgs]))
    
    # print the allocation information
    print("The allocation of dataset")
    for i in range (num_clients):               
       print(len(dict_groups[i]),end=",")
       if((i+1)%10==0):
        print()
  
    return dict_groups

def Cmnist_noniid_unequal(dataset, num_clients):
   
    print('type:non_iid_uneuqal')
    # 12000 images: 1200(num_fragments) × 10(group_imgs), each of user: 2×group_imgs
    num_fragments, frag_imgs = 1200, 10

    #idx_fragments_list: the index list of fragments
    idx_fragments_list = [i for i in range(num_fragments)]
    dict_groups = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(len(dataset))
    
    # get the label list and add the index to it
    indexs_labels = np.arange(len(dataset))
    labels_list=[]
    for i in range(len(dataset)):
      labels_list.append(dataset[i][1])

    # sort labels
    idxs_labels_np = np.vstack((idxs, indexs_labels))
    idxs_labels_np = idxs_labels_np[:, idxs_labels_np[1, :].argsort()]
    indexs_labels = idxs_labels_np[0, :]

    # Minimum and maximum shards assigned per client:
    min_fragments = 5
    max_fragments = 30

    # Every client gets the random amount of fragments
    random_set_np = np.random.randint(min_fragments, max_fragments+1,size=num_clients)
    random_set_size = np.around((random_set_np/sum(random_set_np))*num_fragments).astype(int)
    sum_random_set = sum(random_set_size)
    
    # Ensure that the number of fragments being split is equal to num_fragments
    random_difference = sum_random_set - num_fragments
    if random_difference > 0:
      for i in range(random_difference):
        random_set_size[i]=random_set_size[i]-1
    else:
      for i in range(-random_difference):
        random_set_size[i]=random_set_size[i]+1
    print(random_set_size)

    # devide the dataset ramdomly as the random_set_size
    for i in range(num_clients):
      
      random_set = set(np.random.choice(idx_fragments_list,random_set_size[i],replace=False))
      idx_fragments_list = list(set(idx_fragments_list) - random_set)
      for s in random_set:
        dict_groups[i] = np.concatenate(
                 (dict_groups[i], indexs_labels[s*frag_imgs:(s+1)*frag_imgs]),
                  axis=0)
    
    print("The allocation of dataset")                
    for i in range (num_clients):
      print(len(dict_groups[i]),end=",") 
      if((i+1)%10==0):
        print()       
    print()
    return dict_groups

