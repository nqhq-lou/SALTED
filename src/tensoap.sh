# $1 == $TENSOAP_FILE_IN 
# $2 == parallel task index

for (( lam=0; lam<=$lmax; lam++ )); do

if [ $TENSOAP_NC -ne 0 ]; then
#RUN TO SELECT nc SPARSE FEATURES FROM ns RANDOM STRUCTURES
get_power_spectrum.py -f $1 -lm ${lam} $TENSOAP_P $TENSOAP_VF -s $TENSOAP_SPECIES -c $TENSOAP_SPECIES -nc $TENSOAP_NC -ns $TENSOAP_NS -sm 'random' -o $TENSOAP_OUTDIR/$2FEAT-${lam} -rc $TENSOAP_RC -sg $TENSOAP_SG $TENSOAP_D

#RUN TO COMPUTE FEATURES WITH SPARSE FEATURES PRESELECTED
get_power_spectrum.py -f $1 -lm ${lam} $TENSOAP_P $TENSOAP_VF -s $TENSOAP_SPECIES -c $TENSOAP_SPECIES -sf $TENSOAP_OUTDIR/FEAT-${lam} -o $TENSOAP_OUTDIR/$2FEAT-${lam} -rc $TENSOAP_RC -sg $TENSOAP_SG $TENSOAP_D

else

get_power_spectrum.py -f $1 -lm ${lam} $TENSOAP_P $TENSOAP_VF -s $TENSOAP_SPECIES -c $TENSOAP_SPECIES -o $TENSOAP_OUTDIR/$2FEAT-${lam} -rc $TENSOAP_RC -sg $TENSOAP_SG $TENSOAP_D

fi

done
