#!/usr/bin/perl
#
# Copyright 2018   Suwon Shon
#           2018   Jerry Peng
# Usage: make_voxceleb1_sv.pl /voxceleb1/ data/.

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-voxceleb1-dir> <path-to-output>\n";
  print STDERR "e.g. $0 /voxceleb1/ data\n";
  exit(1);
}
($db_base, $out_base_dir) = @ARGV;


sub trim($)
{
  my $string = shift;
  $string =~ s/^\s+//;
  $string =~ s/\s+$//;
  return $string;
}


# Format file veri_test.txt, output to voxceleb1_trials_sv
$out_dir = "$out_base_dir";

open(IN_TRIALS, "<", "$db_base/veri_test.txt") or die "Count not open the input file $db_base/veri_test.txt";
open(OUT_TRIALS,">", "$out_dir/voxceleb1_trials_sv") or die "Could not open the output file $out_dir/voxceleb1_trials_sv";
# $dummy = <IN_TRIALS>;
while(<IN_TRIALS>) {
  chomp;
  # trim($_);
  $_ =~ s:/:-:g;
  $_ =~ s:\.wav::g;
  ($is_target,$enrollment,$test) = split(" ", $_);
  $target='nontarget';
  if ($is_target eq 1) {
    $target='target';
  }
  print OUT_TRIALS "$enrollment $test $target\n";

}
close(IN_TRIALS) || die;
close(OUT_TRIALS) || die;


@dirs = ("train", "test");
for my $dir (@dirs){
  $num_errs = 0;
  system("mkdir -p $out_dir/$dir") and 
    die "Could not create file $out_dir/$dir";
  # Create wav.scp, utt2psk, spk2utt
  system("find $db_base/$dir -type f -name '*.wav' > $out_dir/$dir/wav.list") and 
    die "Could not create file $out_dir/$dir/wav.list";
  open(WAV_LIST, "<", "$out_dir/$dir/wav.list") or die "Could not open input file $out_dir/$dir/wav.list";
  open(WAV_SCP, ">", "$out_dir/$dir/wav.scp") or die "Could not open output file $out_dir/$dir/wav.scp";
  open(UTT2SPK, ">", "$out_dir/$dir/utt2spk") or die "Could not open output file $out_dir/$dir/wav.scp";

  while(my $line = <WAV_LIST>) {
    chomp $line;
    $line = trim($line);
    if ($line =~ /\/(id\d{5})\/(.*)\/(\d{5})\.wav$/) {
      print WAV_SCP "$1-$2-$3 $line\n";
      print UTT2SPK "$1-$2-$3 $1\n";
    } else {
      $num_errs = $num_errs + 1;
    }
  }
  print "$num_errs audios are abandoned from $dir/wav.list. The rest is saved into $dir/wav.scp\n";

  close(WAV_SCP) || die;
  close(WAV_LIST) || die;
  close(UTT2SPK) || die;


  system("utils/utt2spk_to_spk2utt.pl $out_dir/$dir/utt2spk > $out_dir/$dir/spk2utt") &&
    die "Error creating spk2utt file in directory $out_dir/$dir";
  system("utils/fix_data_dir.sh $out_dir/$dir");
  system("utils/validate_data_dir.sh --no-text --no-feats $out_dir/$dir") &&
    die "Error validating directory $out_dir/$dir";
}
