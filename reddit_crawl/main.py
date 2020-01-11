import praw
import time
import pymysql
from datetime import datetime
import sys
import urllib


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def internet_on():
    try:
        urllib.request.urlopen("https://www.google.com/", timeout=1)
        return True
    except urllib.request.URLError:
        return False

CLIENT_ID = 'xxxxx'
CLIENT_SECRET = 'xxxxxx'
PASSWORD = 'xxxxxx'
USERNAME = 'xxxxxxx'

reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET,password=PASSWORD, user_agent='testscript by /u/mallik1055',username=USERNAME)
#check if authentication is OK
print(reddit.user.me())
print("Reddit API authenticated")
#Check DB
mydb = pymysql.connect(
  host="localhost",
  user="root",
  passwd='root',
  database="nlp_final",
  port=8889,
  autocommit=True,
  charset="utf8mb4"
)
mycursor = mydb.cursor()
print("DB connection established")
#insert the data into DB
query_a = "INSERT IGNORE INTO redditors (user_id,name,comment_karma,link_karma,source_subreddit,acct_created_at,inserted_at) VALUES (%s, %s, %s, %s, %s, %s,%s)"
query_b = "INSERT IGNORE INTO redditor_comments (user_id,comment_id,subreddit,commented_at,inserted_at) VALUES (%s, %s, %s, %s, %s)"
query_c = "INSERT IGNORE INTO comments_master(comment_id,comment_text) VALUES (%s,%s)"




def getChunks(data):

    per_chunk = 1000
    data_chunks = [data[i * per_chunk:(i + 1) * per_chunk] for i in range((len(data) + per_chunk - 1) // per_chunk )]  
    return data_chunks



redditor_data = []
redditor_comments = []
comments_master = []


#all subreddits to scrape
subreddit_name_list = {
    "bipolar2":10000
    #"Discussion":1000,
    #"AskReddit":7000
    #"popular":5000
}

for subreddit_name,limit_count in subreddit_name_list.items():

    print("Running for subreddit = "+subreddit_name)

    bipolar_sr = reddit.subreddit(subreddit_name)

    top_comments = bipolar_sr.top(limit=limit_count)

    for submission in top_comments:

        redditor_data = []
        redditor_comments = []
        comments_master = []
        
        #skip if post does not have author
        if not submission.author:
            continue

        redditor = submission.author

        #skip suspended accts
        if hasattr(redditor, 'is_suspended') and redditor.is_suspended:
            continue
        
        print("Getting results for "+redditor.name)
        
        acct_created_at = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(redditor.created_utc))
        
        inserted_at = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        
        redditor_data+=[(redditor.id,redditor.name,redditor.comment_karma,redditor.link_karma,subreddit_name,acct_created_at,inserted_at)]
        
        res = mycursor.executemany(query_a, redditor_data)
        if not res:
            #user already exists
            #not need to fetch comments
            print("Skipping::User already exists")
            continue

        #search comments by the user
        new_comments = redditor.comments.new(limit=None)
        
        for comment in new_comments:
            commented_at = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(comment.created_utc))
            redditor_comments += [(redditor.id,comment.id,comment.subreddit.display_name,commented_at,inserted_at)]
            comments_master += [(comment.id,comment.body)]
        for data in getChunks(redditor_comments):
            mycursor.executemany(query_b, data)
        for data in getChunks(comments_master):
            mycursor.executemany(query_c, data)
