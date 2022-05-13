#!/usr/bin/env perl

#Author: Xingfu Wu
#MCS, ANL
# exe.pl: average the execution time in 5 runs
#
use Time::HiRes qw(gettimeofday);

foreach $filename (@ARGV) {
 #  print "Start to preprocess ", $filename, "...\n";
   $ssum = 0.0;
   $nmax = 1;
   @nn = (1..$nmax);
   for(@nn) {
    $retval = gettimeofday( );
    # 8 MPI ranks total, 8 ranks per node (1 node)
    system("aprun -n 16 -N 1 $filename >/dev/null 2>&1");
    $tt = gettimeofday( );
    $ttotal = $tt - $retval;
    $ssum = $ssum + $ttotal;
   }
   $avg = $ssum / $nmax;
 #  print "End to preprocess ", $avg, "...\n";
   printf("%.3f", $avg);
}
