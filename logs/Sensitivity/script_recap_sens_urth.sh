#!/bin/sh

for myfile in *"urthead"*
do
  #echo "$myfile"
  grep -wn 'testing FastSlow+URT' $myfile | cut -d: -f1  > tf1
  # echo $myfile
  # peinr
  echo $myfile > t
  echo " " >> t
  while read p; do
    # echo "$p"
    linenumber=$p
    # echo "$linenumber"
    slinenumber=$(($linenumber+1))
    elinenumber=$(($linenumber+3))
    
    sed -n "$slinenumber","$elinenumber"p $myfile > tmp
    
    awk '/mse/{print}' tmp > tmp1
    awk '{gsub(/[a-z]/, "");print}' tmp1 > tmp2
    awk '{gsub(/:/, "");print}' tmp2 > tmp3
    awk '{gsub(/" "/, "");print}' tmp3 > tmp4
    awk '{gsub(/','/, "");print}' tmp4 > tmp5

    awk '//{print $1 "," $2}' tmp5 >> t

  done < tf1
  
  # echo "" >> t
  
  # while read p; do
  #   # echo "$p"
  #   linenumber=$p
  #   # echo "$linenumber"
  #   slinenumber=$(($linenumber+1))
  #   elinenumber=$(($linenumber+3))
    
  #   sed -n "$slinenumber","$elinenumber"p $myfile > tmp
    
  #   awk '/mse/{print}' tmp > tmp1
  #   awk '{gsub(/[a-z]/, "");print}' tmp1 > tmp2
  #   awk '{gsub(/:/, "");print}' tmp2 > tmp3
  #   awk '{gsub(/" "/, "");print}' tmp3 > tmp4
  #   awk '{gsub(/","/, "");print}' tmp4 > tmp5

  #   awk '//{print $2}' tmp5 >> t

  # done < tf1

  awk '
  { 
    for (i=1; i<=NF; i++)  {
      a[NR,i] = $i
    }
  }
  NF>p { p = NF }
  END {    
      for(j=1; j<=p; j++) {
    str=a[1,j]
    for(i=2; i<=NR; i++){
        str=str" "a[i,j];
    }
    print str
      }
  }' t > t1
  
  
  awk '{gsub(/ /, ",");print}' t1

#   linenumber=$(cat tmp)
#   #echo "linenumber: " $linenumber
#   slinenumber=$(($linenumber+1))
#   elinenumber=$(($linenumber+3))
  
#   #tail -n 51 $myfile > tmp
#   #head -n 49 tmp > tmp2
#   #mstr=`awk '{print $12}' tmp2` 
#   #echo $myfile 
#   #echo $myfile $slinenumber $elinenumber
  
#   sed -n "$linenumber","$elinenumber"p $myfile > tmp
  
#   awk '{gsub(/\+\-/, " ");print}' tmp > tmp1
#   awk '{gsub(/\]|\[/, "");print}' tmp1 > tmp2
#   awk '{gsub(/acc/, "");print}' tmp2 > tmp3
  
  
#   echo $myfile > t
#   echo " " >> t
#   awk '/Session/{print $6}' tmp3 >> t
#   echo "" >> t
#   awk '/novel/{print ($7*100)}' tmp3 >> t
#   echo "" >> t
#   awk '/Session/{print $7}' tmp3 >> t
  

#   #Invert the entry
#   awk '
# 	{ 
# 	    for (i=1; i<=NF; i++)  {
# 		a[NR,i] = $i
# 	    }
# 	}
# 	NF>p { p = NF }
# 	END {    
# 	    for(j=1; j<=p; j++) {
# 		str=a[1,j]
# 		for(i=2; i<=NR; i++){
# 		    str=str" "a[i,j];
# 		}
# 		print str
# 	    }
# 	}' t > t1
	
	
#   awk '{gsub(/ /, ",");print}' t1

done


