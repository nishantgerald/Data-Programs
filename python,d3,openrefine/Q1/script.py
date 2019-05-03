#!/usr/bin/env python3
import http.client
import json
import time
import sys
import collections
startTime = time.time()
conn = http.client.HTTPSConnection("api.themoviedb.org")
payload = "{}"
request_counter=0
max_results=50
movie_limit=350
sleep_time=0.25
api_key=sys.argv[1]
#api_key=1844bd7c9c50b69bdc9c536a0a525e86
#url=/3/discover/movie?with_genres=35&primary_release_year=2004&page=1&language=en-US&api_key=&sort_by=popularity.desc
for i in range(1,max_results):
	url="/3/discover/movie?with_genres=18&primary_release_date.gte=2004-01-01&page="+str(i)+"&items_per_page=10&language=en-US&api_key="+api_key+"&sort_by=popularity.desc"
	conn.request("GET", url, payload)
	res = conn.getresponse()
	data = res.read()
	results=json.loads(data)
	#print(data.decode("utf-8"))
	#for i in info:
	#	print(i,end='')
	with open('movie_ID_name.csv','a') as f:
		for movie in results['results']:
			request_counter+=1
			if (request_counter>movie_limit):
				break
			print(str(movie['id'])+','+movie['title'],file=f)
	if (request_counter>movie_limit):
		break

with open('movie_ID_name.csv','r') as f1:
	sim_list=[]
	for line in f1:
		request_counter=0
		current_movie_id=line.strip().split(',',maxsplit=1)[0]
		current_movie_name=line.strip().split(',',maxsplit=1)[1]
		url= '/3/movie/'+current_movie_id+"/similar_movies?api_key="+api_key
		conn.request("GET", url, payload)
		res = conn.getresponse()
		data = res.read()
		results=json.loads(data)
		for movie in results['results']:
			request_counter+=1
			if request_counter>5:
				break
			#print(str(movie['id']),movie['title'])
			if ([movie['id'],current_movie_id] not in sim_list):
				sim_list.append([current_movie_id,movie['id']])
		time.sleep(sleep_time)
	with open('movie_ID_sim_movie_ID.csv','a') as f2:
		for sim_pair in sim_list:
			print(str(sim_pair[0])+','+str(sim_pair[1]),file=f2)
print ('Time for script to run: {0} seconds'.format(time.time() - startTime))
