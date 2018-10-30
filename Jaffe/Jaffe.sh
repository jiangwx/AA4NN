for img in $(ls *.jpg)
do
	./Jaffe $img ./mean.jpg
done
