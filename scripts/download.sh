SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

mkdir $SCRIPTPATH/../data
cd $SCRIPTPATH/../data
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip 
mv shapenetcore_partanno_segmentation_benchmark_v0 shapenet
rm shapenetcore_partanno_segmentation_benchmark_v0.zip
cd -
