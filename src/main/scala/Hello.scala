import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.regression.LabeledPoint
    
object Hello {
  def main(args: Array[String]): Unit = {
        val conf = new SparkConf().setMaster("local").setAppName("hello")
        val sc = new SparkContext(conf)
        val sqlContext = new org.apache.spark.sql.SQLContext(sc)
        println("Hello, world!")
        //val df = sqlContext.read.json("test.json")
        var df = sqlContext.read
        .format("com.databricks.spark.csv")
        .option("header", "true") // Use first line of all files as header
        .load("articlee.csv")
        val data = df.select("original_id", "title", "label").toDF("id", "text", "label")
        val splits = data.randomSplit(Array(0.6, 0.4))
        val training = splits(0).cache()
        val test = splits(1)
        val tokenizer = new Tokenizer()
        .setInputCol("text")
        .setOutputCol("words")
        val tokenized = tokenizer.transform(training.select("text"))
        tokenized.toString()
        tokenized.collect().foreach(println) */
        val hashingTF = new HashingTF()
        .setInputCol(tokenizer.getOutputCol)
        .setOutputCol("features")
        val hashing = hashingTF.transform(tokenized)
    //    hashing.collect().foreach(println)
        val idf = new IDF().setInputCol("features").setOutputCol("newfeatures")
        val idfModel = idf.fit(hashing)
        val tfidf = idfModel.transform(hashing)
        var fiturtraining = tfidf.select("newfeatures")
        val labeltraining = training.select("label")
        fiturtraining.join(training.select("label"))
        val ftraining = labeltraining.join(fiturtraining)
        fiturtraining.collect().foreach(println)
        val svm = new SVMWithSGD()  
        val labeled = ftraining.map(row => LabeledPoint(row.getDouble(0), row(1).asInstanceOf[Vector]))
        val model = svm.run(labeled)
     /*   val a = tfidf.rdd
        val featuresDF = tfidf.select("newfeatures")
        val features: RDD[Vector] = featuresDF.map { case Row(v: Vector) => v }
        val kmeans = new KMeans()
        val model = kmeans.run(features) 
        val kmeans = new KMeans()
      .setK(2)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      println(kmeans.explainParams())
      tfidf.cache()
      val model = kmeans.fit(tfidf) */
      
      }
    }