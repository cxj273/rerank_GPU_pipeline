#!/usr/bin/perl

use strict;
use warnings;

if( scalar(@ARGV) != 2 ){
    print STDERR "usage: perl select_files.pl input_file_list video_idx_list > selected_output_video_list\n";
    exit -1;
}

my %mm = ();
open(AA, $ARGV[0]);
while(<AA>){
    chomp;
    my @arr = split / /, $_;
    my $path = $arr[0];
    my $name = $arr[0];
    $name =~ s/.*\/HVC//;
    $name =~ s/\..*//;
    $mm{$name} = $path;
}
close(AA);

my $found = 0;
my $lines = 0;
open(AA, $ARGV[1]);
while(<AA>){
    chomp;
    $lines++;
    if( ! defined $mm{$_} ){
        print STDERR "feature file for $_ not found\n";
    }else{
        print $mm{$_}."\n";
        $found++;
    }
}
close(AA);

print STDERR "found $found/$lines features\n";
