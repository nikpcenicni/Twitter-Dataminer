import tweepy
import pandas as pd
from queue import Queue
import os
from dotenv import load_dotenv

q = Queue()

load_dotenv()

auth = tweepy.OAuthHandler(os.getenv("CONSUMER_KEY"), os.getenv("CONSUMER_SECRET"))
auth.set_access_token(os.getenv("ACCESS_TOKEN"), os.getenv("ACCESS_TOKEN_SECRET"))
api = tweepy.API(auth, wait_on_rate_limit=True)

# Output details
out_user = 'users.csv'
out_followers = 'followers.csv'
out_skip = 'skipped.csv'

def build_network(username):
    # Setting up data frames with initial user
    user = api.get_user(screen_name=username)
    user_details = {'name':user.name,
        'screen_name':user.screen_name,
        'created_at':str(user.created_at),
        'id':user.id,
        'friends_count':user.friends_count,
        'followers_count':user.followers_count}
    master_followers = pd.DataFrame({'id':user.id,'followers':api.get_follower_ids(user_id=user.id)})
    master_user_details = pd.DataFrame([user_details])

    # Setting up skip_listlist
    skip_list = []
    skip_list.append(user_details['id'])
    skip_df = pd.DataFrame({'id':user.id},index=[0])

    # Exporting initial master files
    master_user_details.to_csv(out_user,index=False)
    master_followers.to_csv(out_followers,index=False)
    skip_df.to_csv(out_skip,index=False)

    # Putting initial follower seed in queue
    list(map(q.put,master_followers['followers']))
    print (len(list(q.queue)))

    while not q.empty():
        u = q.get()
        if u in skip_list:
            continue
        else:
                try:
                    # API call to get user data
                    user = api.get_user(user_id=u)
                    user_details = {'name':user.name,
                                    'screen_name':user.screen_name,
                                    'created_at':str(user.created_at),
                                    'id':user.id,
                                    'friends_count':user.friends_count,
                                    'followers_count':user.followers_count}
                    
                    # Adding to skip list
                    skip_list.append(user_details['id'])
                    skip_df = pd.DataFrame({'id':user.id},index=[0])
                    
                    # Appending user data to master list
                    user_details = pd.DataFrame([user_details])
                    master_user_details = master_user_details.append(user_details)
                    
                    # Getting followers and appending to master list
                    followers = pd.DataFrame({'id':user.id,'followers':api.get_follower_ids(user_id=user.id)})
                    if followers.shape[0] > 200:
                        followers = followers.sample(200)
                    else:
                        pass
                    master_followers = master_followers.append(followers)
                    
                    # Adding retrieved followers to queue
                    list(map(q.put,followers['followers']))
                                
                    # Exporting user and followers to CSV
                    user_details.to_csv(out_user,index=False,mode='a',header=False)
                    followers.to_csv(out_followers,index=False,mode='a',header=False)
                    skip_df.to_csv(out_skip,index=False,mode='a',header=False)
                    
                    print (len(list(q.queue)))
                    
                # Error handling
                except tweepy.TweepError as error:
                    print (type(error))
            
                    if str(error) == 'Not authorized.':
                        print ('Can''t access user data - not authorized.')
                        skip_list.append(u)
                        skip_df = pd.DataFrame({'id':u},index=[0])
                        skip_df.to_csv(out_skip,index=False,mode='a',header=False)
            
                    if str(error) == 'User has been suspended.':
                        print ('User suspended.')
                        skip_list.append(u)
                        skip_df = pd.DataFrame({'id':u},index=[0])
                        skip_df.to_csv(out_skip,index=False,mode='a',header=False)    


def resume_network(username):
    # Setting up data frames with initial
    master_user_details = pd.read_csv(out_user)
    master_followers = pd.read_csv(out_followers)
    skip_df = pd.read_csv(out_skip)

    skip_list = []
    skip_list = list(skip_df['id'])

    last_id = master_user_details['id'].tail(1)
    last_id = last_id.iloc[0]
    last_id_idx = master_followers[master_followers['followers'] == last_id]
    last_id_idx = last_id_idx.head(1)
    last_id_idx = last_id_idx.index.values[0]

    queue_list = list(master_followers['followers'].iloc[(last_id_idx+1):,])

    list(map(q.put,queue_list))
    print(len(list(q.queue)))

    while not q.empty():
        u = q.get()
        if u in skip_list:
            continue
        else:
                try:
                    # API call to get user data
                    user = api.get_user(user_id=u)
                    user_details = {'name':user.name,
                                    'screen_name':user.screen_name,
                                    'created_at':str(user.created_at),
                                    'id':user.id,
                                    'friends_count':user.friends_count,
                                    'followers_count':user.followers_count}
                    
                    # Adding to skip list
                    skip_list.append(user_details['id'])
                    skip_df = pd.DataFrame({'id':user.id},index=[0])
                    
                    # Appending user data to master list
                    user_details = pd.DataFrame([user_details])
                    master_user_details = master_user_details.append(user_details)
                    
                    # Getting followers and appending to master list
                    followers = pd.DataFrame({'id':user.id,'followers':api.get_follower_ids(user_id=user.id)})
                    if followers.shape[0] > 200:
                        followers = followers.sample(200)
                    else:
                        pass
                    master_followers = master_followers.append(followers)
                    
                    # Adding retrieved followers to queue
                    list(map(q.put,followers['followers']))
                                
                    # Exporting user and followers to CSV
                    user_details.to_csv(out_user,index=False,mode='a',header=False)
                    followers.to_csv(out_followers,index=False,mode='a',header=False)
                    skip_df.to_csv(out_skip,index=False,mode='a',header=False)
                    
                    print (len(list(q.queue)))
                    
                # Error handling
                except tweepy.TweepError as error:
                    print (type(error))
            
                    if str(error) == 'Not authorized.':
                        print ('Can''t access user data - not authorized.')
                        skip_list.append(u)
                        skip_df = pd.DataFrame({'id':u},index=[0])
                        skip_df.to_csv(out_skip,index=False,mode='a',header=False)
            
                    if str(error) == 'User has been suspended.':
                        print ('User suspended.')
                        skip_list.append(u)
                        skip_df = pd.DataFrame({'id':u},index=[0])
                        skip_df.to_csv(out_skip,index=False,mode='a',header=False)    






build_network('nikpcenicni')
#resume_network('elonmusk')