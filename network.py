import tweepy
import pandas as pd
from queue import Queue
import os
from dotenv import load_dotenv
from jaal import Jaal
from jaal.datasets import load_got
from dash import html
import networkx as nx
from matplotlib.pyplot import figure
import plotly.graph_objects as go


q = Queue()

load_dotenv()

auth = tweepy.OAuthHandler(os.getenv("CONSUMER_KEY"), os.getenv("CONSUMER_SECRET"))
auth.set_access_token(os.getenv("ACCESS_TOKEN"), os.getenv("ACCESS_TOKEN_SECRET"))
api = tweepy.API(auth, wait_on_rate_limit=True)

# Output details

def set_globals(type):
    if type == "offensive":
        out_user = 'Datasets/users_offensive.csv'
        out_followers = 'Datasets/followers_offensive.csv'
        out_skip = 'Datasets/skipped_offensive.csv'
        return out_user, out_followers, out_skip
    elif type == "hate":
        out_user = 'Datasets/users_hate.csv'
        out_followers = 'Datasets/followers_hate.csv'
        out_skip = 'Datasets/skipped_hate.csv'
        return out_user, out_followers, out_skip
    elif type == "test":
        out_user = 'Datasets/users.csv'
        out_followers = 'Datasets/followers.csv'
        out_skip = 'Datasets/skipped.csv'
        return out_user, out_followers, out_skip
                     
def build_network(username, type, length):
    out_user, out_followers, out_skip = set_globals(type)
    # Setting up data frames with initial user
    user = api.get_user(screen_name=username)
    user_details = {'name':user.name,
        'screen_name':user.screen_name,
        'created_at':str(user.created_at),
        'id':user.id,
        'friends_count':user.friends_count,
        'followers_count':user.followers_count}
    followers = pd.DataFrame({'to':user.id,'from':api.get_follower_ids(user_id=user.id),"username":user.screen_name})
    if followers.shape[0] > length:
        followers = followers.sample(length)
        master_followers = followers
    else:
        master_followers = followers
        pass
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
    list(map(q.put,master_followers['from']))
    print (len(list(q.queue)))

    while not q.empty() and len(master_user_details) < length*length:
        u = q.get()
        if u in skip_list:
            continue
        elif len(skip_list) >= length:
            break
        else:
            try:
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
                    followers = pd.DataFrame({'to':user.id,'from':api.get_follower_ids(user_id=user.id),"username":user.screen_name})
                    if followers.shape[0] > length:
                        followers = followers.sample(length)
                    else:
                        pass
                    master_followers = master_followers.append(followers)
                    
                    # Adding retrieved followers to queue
                    list(map(q.put,followers['to']))
                                
                    # Exporting user and followers to CSV
                    user_details.to_csv(out_user,index=False,mode='a',header=False)
                    master_followers.to_csv(out_followers,index=False,mode='a',header=False)
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
            except Exception as e:
                print ('Error: ',e)
                continue
    return master_followers, master_user_details

def view_network(type):
    edge = ''
    node = ''
    if type == 'hate':
        path = 'Datasets/hate_followers.csv'
        node = 'Datasets/hate_user_details.csv'
    elif type == 'offensive':
        path = 'Datasets/offensive_followers.csv'
        node = 'Datasets/offensive_user_details.csv'
    # load the data
    edge_df = pd.read_csv(path)
    node_df = pd.read_csv(node)
    Jaal(edge_df,node_df).plot()



# mostf, mostu = build_network('vinoo_96',"hate", 3)
# print(mostf.head())
# print(mostu.head())
# # #resume_network('elonmusk')
#view_network("hate")
