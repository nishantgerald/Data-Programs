// Databricks notebook source
// MAGIC %md
// MAGIC #### Q2 - Skeleton Scala Notebook
// MAGIC This template Scala Notebook is provided to provide a basic setup for reading in / writing out the graph file and help you get started with Scala.  Clicking 'Run All' above will execute all commands in the notebook and output a file 'examplegraph.csv'.  See assignment instructions on how to to retrieve this file. You may modify the notebook below the 'Cmd2' block as necessary.
// MAGIC 
// MAGIC #### Precedence of Instruction
// MAGIC The examples provided herein are intended to be more didactic in nature to get you up to speed w/ Scala.  However, should the HW assignment instructions diverge from the content in this notebook, by incident of revision or otherwise, the HW assignment instructions shall always take precedence.  Do not rely solely on the instructions within this notebook as the final authority of the requisite deliverables prior to submitting this assignment.  Usage of this notebook implicitly guarantees that you understand the risks of using this template code. 

// COMMAND ----------

/*
DO NOT MODIFY THIS BLOCK
This assignment can be completely accomplished with the following includes and case class.
Do not modify the %language prefixes, only use Scala code within this notebook.  The auto-grader will check for instances of <%some-other-lang>, e.g., %python
*/
import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.functions._
case class edges(Source: String, Target: String, Weight: Int)
import spark.implicits._

// COMMAND ----------

/* 
Create an RDD of graph objects from our toygraph.csv file, convert it to a Dataframe
Replace the 'examplegraph.csv' below with the name of Q2 graph file.
*/

val df = spark.read.textFile("/FileStore/tables/bitcoinotc.csv") 
  .map(_.split(","))
  .map(columns => edges(columns(0), columns(1), columns(2).toInt)).toDF()

// COMMAND ----------

// Insert blocks as needed to further process your graph, the division and number of code blocks is at your discretion.

// COMMAND ----------

// Eliminating duplicate rows
val distinct_df = df.dropDuplicates()

// COMMAND ----------

//Filtering  nodes by edge weight >= 5
val filter_df = distinct_df.filter(distinct_df("weight") >= 5)

// COMMAND ----------

val outDegree_df=filter_df.groupBy(filter_df("source") as "node").agg(sum("weight").alias("weighted-out-degree")).sort(desc("weighted-out-degree"),asc("node"))
val inDegree_df=filter_df.groupBy(filter_df("target") as "node").agg(sum("weight").alias("weighted-in-degree")).sort(desc("weighted-in-degree"),asc("node"))
val totDegree_df_lim = outDegree_df.join(inDegree_df,Seq("node"),"outer").withColumn("weighted-total-degree", $"weighted-in-degree"+$"weighted-out-degree").sort(desc("weighted-total-degree"),asc("node")).drop("weighted-in-degree","weighted-out-degree").limit(1)
val outDegree_df_lim=outDegree_df.limit(1)
val inDegree_df_lim=inDegree_df.limit(1)

// COMMAND ----------

val i_node=inDegree_df_lim.first().getString(0)
val i_degree=inDegree_df_lim.first().getLong(1)
val o_node=outDegree_df_lim.first().getString(0)
val o_degree=outDegree_df_lim.first().getLong(1)
val t_node=totDegree_df_lim.first().getString(0)
val t_degree=totDegree_df_lim.first().getLong(1)
val union=Seq((i_node,i_degree,"i"),
              (o_node,o_degree,"o"),
              (t_node,t_degree,"t")).toDF("v","d","c")
display(union)
