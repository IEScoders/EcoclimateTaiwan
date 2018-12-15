ls *.html | while read file
do
sed -i '' '/'bootstrap.min.css'/d' $file
sed -i '' '/'bootstrap-theme.min.css'/d' $file
sed -i '' '/'100%'/d' $file
done

