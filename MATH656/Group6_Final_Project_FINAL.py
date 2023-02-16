'''
Python Final Project
'''

#download packages
import pandas as pd
from nltk import word_tokenize
from nltk import bigrams, trigrams
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter


#download data
df=pd.read_csv(r"/Users/rogerlo/Desktop/Math656_Data_Mining/Final/mypersonality.csv",encoding='cp1252')

#show top 5 rows of data
df.head()

#perform cleaning by removing extraneous columns
df=df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4','Unnamed: 5', 'Unnamed: 6'],axis=1)

#remove blank rows (of which there is only 1)
df.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)

df = df.drop(labels=9419)    # Delete row with missing status

#rename df column
df.rename(columns={'#AUTHID':'id'},inplace=True)

#describe characteristics
df.describe()

#cleaned dataset to csv
#df.to_csv(r"C:\Users\lahenderson\Documents\Gtown - MAST\656\Project_Final\data\cleaned_data\final_cleaned_df.csv")
df.to_csv(r"/Users/rogerlo/Desktop/Math656_Data_Mining/Final/clean_mypersonality.csv")


#practice unigram
text="The quick brown fox jumps over the lazy dog"
unigrams=word_tokenize(text)
print(unigrams)

#new column "tokenized_status" #unigram/bag of words
tt=TweetTokenizer()
df['tokenized_status']=df['STATUS'].apply(tt.tokenize)
df['standard_unigram_status']=df['STATUS'].apply(word_tokenize)

#total number of words per social media post
df['word_count_per_post']=df['tokenized_status'].apply(lambda x: len(x))

#total posts (shy person= few posts, sociable= many posts)
df['count_by_id'] = df['id'].groupby(df['id']).transform('count')
print(df)

#create sets of words
thunder_set={'thunder'} #example
ego_set={'i','me','myself','we'} #list of egocentric words
social_set={'!','<3',':D',':)','party','friend','friends','friendship'} #social set
agreable_set={'compromise','agree','agreement','agreed'} #agreable set

#function to count terms
def term_count(input_list, comparison_set):
    return len([1 for word in input_list if str(word).lower() in comparison_set]) #1 is a placeholder, it could be anything

#create columns with special sets
df['count_egoism']=df['tokenized_status'].apply(lambda x: term_count(x,ego_set)) #egoism
df['count_social']=df['tokenized_status'].apply(lambda x: term_count(x,social_set)) #social
df['count_agreable']=df['tokenized_status'].apply(lambda x: term_count(x,agreable_set)) #agreable
#df['thunder_test']=df['tokenized_status'].apply(lambda x: term_count(x,thunder_set))

#subset
test_df=df[['standard_unigram_status','tokenized_status','count_egoism','count_social','count_agreable']]
test_df

# file output to csv
df.to_csv(r"/Users/rogerlo/Desktop/Math656_Data_Mining/Final/final_output_mypersonality.csv")

#import packages for PCA & K-means
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#confirm no more N/As?
print('Is there any missing value? ', df.isnull().values.any())
print('How many missing values? ', df.isnull().values.sum())
df.dropna(inplace=True)
print('Number of participants after eliminating missing values: ', len(df))


#create id df, sort by id, & de-duplicate for merging later
df_id=df[['id']].copy()
df_id=df_id.sort_values(by=['id'])
df_id.drop_duplicates(keep='first',inplace=True)
df_id

#sort values & change yes to 1, no to 0
df=df.sort_values(by=['id'])
df['cEXT'] = df['cEXT'].map({'y': 1, 'n': 0})
df['cNEU'] = df['cNEU'].map({'y': 1, 'n': 0})
df['cAGR'] = df['cAGR'].map({'y': 1, 'n': 0})
df['cCON'] = df['cCON'].map({'y': 1, 'n': 0})
df['cOPN'] = df['cOPN'].map({'y': 1, 'n': 0})

#group by function and take mean for each personality trait
df=df.groupby(['id']).mean()
df_golden=df.copy()
df_golden=df_golden[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]

#group by person(id) and return mean results for numerical values
df=df[['NETWORKSIZE', 'BETWEENNESS', 'NBETWEENNESS',
       'DENSITY', 'BROKERAGE', 'NBROKERAGE', 'TRANSITIVITY','word_count_per_post', 'count_by_id', 'count_egoism', 'count_social',
       'count_agreable']]



#scale numerical values
columns = list(df.columns)
scaler = MinMaxScaler(feature_range=(0,1)) #scale the data between (0,1)
df = scaler.fit_transform(df)
df = pd.DataFrame(df, columns=columns)

#export for clustering in r
df.to_csv(r'/Users/rogerlo/Desktop/Math656_Data_Mining/Final/clustering_output.csv')

#K means
# Visualize the elbow for K means
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

#instantiate kmeans
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,15))
visualizer.fit(df)
visualizer.poof()


# Creating K-means Cluster Model
from sklearn.cluster import KMeans

# Define 7 clusters and fit my model
kmeans = KMeans(n_clusters=7)
k_fit = kmeans.fit(df)


# Predicting the Clusters
pd.options.display.max_columns = 10
predictions = k_fit.labels_
df['Clusters'] = predictions
df.head()


# How many individuals do we have for each cluster
df.Clusters.value_counts()


### Visulizing the clustering results

# In order to visualize in 2D graph, I will use PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_fit = pca.fit_transform(df)

df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])
df_pca['Clusters'] = predictions
df_pca.head()


plt.figure(figsize=(10,10))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Clusters', palette='Set2', alpha=0.8)
plt.title('Personality Clusters after PCA');

#combine dataframes for comparing clustering results to gold standard
#they are already sorted by prior id and de-duplicated
df['join_id']=range(1,len(df)+1)
df_golden['join_id']=range(1,len(df_golden)+1)
df=df.merge(df_golden,how='inner',on='join_id')

df = df[['NETWORKSIZE', 'BETWEENNESS', 'NBETWEENNESS', 'DENSITY', 'BROKERAGE',
       'NBROKERAGE', 'TRANSITIVITY', 'word_count_per_post', 'count_by_id',
       'count_egoism', 'count_social', 'count_agreable', 'Clusters',
       'cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]

#to csv for analysis
df.to_csv(r'/Users/rogerlo/Desktop/Math656_Data_Mining/Final/clustering_results.csv')
