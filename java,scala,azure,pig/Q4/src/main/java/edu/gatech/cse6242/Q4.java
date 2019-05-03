package edu.gatech.cse6242;

import java.util.StringTokenizer;
import java.lang.Object;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.IOException;

public class Q4 {

  public static class TokenizerMapper
       extends Mapper<Object, Text, IntWritable, IntWritable>
{
    IntWritable out = new IntWritable();
    IntWritable in = new IntWritable();
    private final static IntWritable one = new IntWritable(1);
    private final static IntWritable neg_one = new IntWritable(-1);
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException
    {
        StringTokenizer itr = new StringTokenizer(value.toString(),"\n");

        while (itr.hasMoreTokens()) {

        String line=itr.nextToken();
        String tokens[]=line.split("\t");
        out.set(Integer.parseInt(tokens[0]));
        in.set(Integer.parseInt(tokens[1]));
        context.write(out, one);
        context.write(in,neg_one);
    }
  }
}

public static class TokenizerMapper2
     extends Mapper<Object, Text, IntWritable, IntWritable>
{
    IntWritable deg_diff = new IntWritable();
    private final static IntWritable one = new IntWritable(1);
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException
    {
      StringTokenizer itr = new StringTokenizer(value.toString(),"\n");

      while (itr.hasMoreTokens()) {
          String line=itr.nextToken();
          String tokens[]=line.split("\t");
          deg_diff.set(Integer.parseInt(tokens[1]));
          context.write(deg_diff, one);
    }
  }
}

  public static class IntSumReducer
       extends Reducer<IntWritable,IntWritable,IntWritable,IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(IntWritable key, Iterable<IntWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      int diff = 0;
      for (IntWritable val : values) {
        diff += val.get();
      }
      result.set(diff);
      context.write(key, result);
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "Q4");
    job.setJarByClass(Q4.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path("tempPath"));
    job.waitForCompletion(true);

    Job job2 = Job.getInstance(conf, "Q4");
    job2.setJarByClass(Q4.class);
    job2.setMapperClass(TokenizerMapper2.class);
    job2.setCombinerClass(IntSumReducer.class);
    job2.setReducerClass(IntSumReducer.class);
    job2.setOutputKeyClass(IntWritable.class);
    job2.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job2, new Path("tempPath"));
    FileOutputFormat.setOutputPath(job2, new Path(args[1]));
    System.exit(job2.waitForCompletion(true) ? 0 : 1);
  }
}
