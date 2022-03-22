import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import from_unixtime, to_timestamp, weekofyear, date_format
from pyspark.sql.types import *

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']

def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark

def process_song_data(spark, input_data, output_data):
    """
    Role: the function loads song_data hosted on aws S3 then performs some extraction for certain fields from the data to
    create two tables: 
                       * songs table 
                       * artist tables
    Then it writes the created tables into parquet format that will be uploaded again to aws S3 storage
        
    Parameters:
            spark(str)       : The created instance of the Spark Session
            input_data(str)  : The path to the S3 bucket hosting the song_data
            output_data(str) : The path to the S3 bucket to store the output parquet files
    """

    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"
    
    # read song data file
    df = spark.read.json(song_data)
    
    # creating a temporary table in order to enable the use of sql queries 
    df.createOrReplaceTempView("songs")

    # extract columns to create songs table
    songs_table = df.select('song_id', 'title', 'artist_name', 'artist_id', 'year', 'duration').dropDuplicates(['song_id'])
   
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').parquet(path = output_data + 'songs/songs.parquet', mode ='overwrite')

    # extract columns to create artists table
    artists_table_query = "SELECT distinct artist_id,\
                        artist_name as name,\
                        artist_location as location, \
                        artist_latitude as latitude, \
                        artist_longitude as longitude\
                        FROM songs"
    
    artists_table = spark.sql(artists_table_query)
    
    # write artists table to parquet files
    artists_table = artists_table.write.parquet(path = output_data + "/artists/artists.parquet", mode = "overwrite")

def process_log_data(spark, input_data, output_data):
    """
        Role: the function reads log_data hosted on aws S3 in json formate\
        then performs some extraction for certain fields from the data to
        In addition, it reads song_data datasets and extracts columns for songplays table.
    create three tables: 
                       * users table 
                       * time tables
                       * songplays table
    Then it writes the created tables into parquet format that will be uploaded again to aws S3 storage.
        
    Parameters:
            spark(str)       : The created instance of the Spark Session
            input_data(str)  : The path to the S3 bucket hosting the log_data
            output_data(str) : The path to the S3 bucket to store the output parquet files

    """
    
    
    # get filepath to log data file
    log_data = input_data + 'log_data/*.json'
    
    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df_filtered = df.filter(df.page == "NextSong")

    # extract columns for users table    
    users_table = df.select('userId', 'firstName', 'lastName', 'gender', 'level').dropDuplicates()
    
    ### users_table.createOrReplaceTempView('users')
    
    # write users table to parquet files
    users_table.write.parquet(path = output_data + "/users/users.parquet", mode = "overwrite")

    # create timestamp column from original timestamp column
    df = df.withColumn('timestamp', from_unixtime(df.ts/1000))
    df_filtered = df_filtered.withColumn('timestamp', from_unixtime(df_filtered.ts/1000))
    
    # create datetime column from original timestamp column
    df = df.withColumn('datetime', to_timestamp('timestamp'))
    df_filtered = df_filtered.withColumn('datetime', to_timestamp('timestamp'))
    
    # extract columns to create time table
    time_table = df_filtered.select('datetime', date_format('datetime', 'H').cast(IntegerType()).alias('Hour'),
                 date_format('datetime', 'E').alias('DOW'),
                 date_format('datetime', 'd').cast(IntegerType()).alias('DOM'),
                 date_format('datetime', 'D').cast(IntegerType()).alias('DOY'),
                 date_format('datetime', 'MMMM').alias('Month'),
                 date_format('datetime', 'y').cast(IntegerType()).alias('year')
                 ).withColumn('week', weekofyear('datetime'))\
                  .dropDuplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").parquet(path = output_data + "/time/time.parquet", mode = "overwrite")

    # read in song data to use for songplays table
    song_df = spark.read.parquet(output_data + "/songs/songs.parquet")

    # extract columns from joined song and log datasets to create songplays table 
    joined_df= df_filtered.join(song_df, (df_filtered.artist== song_df.artist_name)\
                             & (df_filtered.song == song_df.title) & (df_filtered.length == song_df.duration), 'left_outer')
    
    songplays_table = joined_df.select(joined_df.timestamp,
            col("userId").alias('user_id'),
            joined_df.level,
            song_df.song_id,
            song_df.artist_id,
            col("sessionId").alias("session_id"),
            joined_df.location,
            col("useragent").alias("user_agent"),
            date_format(joined_df.datetime, 'MMMM').alias('month'),
            date_format(joined_df.datetime, 'y').cast(IntegerType()).alias('year'))\
            .withColumn('songplay_id', monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").parquet(path = output_data + "/songplays/songplays.parquet", mode = "overwrite")

def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
 
    output_data = "s3a://semsem-dend/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)

if __name__ == "__main__":
    main()
