name := """hello-scala"""

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.4" % "test"

libraryDependencies ++= Seq(
  "org.apache.spark"  % "spark-core_2.10"              % "1.6.0",
  "commons-io" % "commons-io" % "2.4",
  "org.apache.spark"  % "spark-mllib_2.10"             % "1.6.0"
  )
  libraryDependencies += "org.apache.spark" %% "spark-sql" % "1.6.0"
  libraryDependencies += "com.databricks" %% "spark-csv" % "1.4.0"
  

fork in run := true