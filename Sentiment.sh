# pick a comment
#

for i in {1..10}
do
	random=$RANDOM
	echo "Random number: $random"

	head -n $random imdb_master-UTF.csv | tail -n 1
	head -n $random imdb_master-UTF.csv | tail -n 1 |  sed 's/.*"\(.*\)".*/\1/' > comment.txt
	echo "Running "
	ollama run llama3.2 "$(cat comment.txt)" classify this comment as positive or negative. do not explain why
	ollama run nemotron-mini "$(cat comment.txt)" classify this comment as positive or negative. do not explain why
done
