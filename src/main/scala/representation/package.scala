package fda

import breeze.linalg.{DenseVector, diag, linspace}
import breeze.stats.distributions.{Gaussian, MultivariateGaussian, Uniform}

import scala.util.Random.nextInt

package object representation {

  /**
   * Helper function to simulate independent sparsely observed functional data.
   * @param domain Domain of the functional data.
   * @param meanFunction Mean function
   * @param eigenFunctions Eigenfunctions generating the data.
   * @param eigenValues Eigenvalues with each function.
   * @param sigma2 Noise variance to add to observed values.
   * @param nSubjects Number of observed functional data.
   * @param nRegular Maximum observations in regular grid.
   * @param sparsity Observe uniformly between (lo, high) observations for each function.
   * @return Functional Data instance.
   */
  def simulateFd( domain: (Double, Double),
                  meanFunction: Double => Double,
                  eigenFunctions: Double => DenseVector[Double],
                  eigenValues: DenseVector[Double],
                  sigma2: Double,
                  nSubjects: Int,
                  nRegular: Int,
                  sparsity: (Int, Int)): Fd ={
    val mvn = MultivariateGaussian(DenseVector(0.0, 0.0), diag(eigenValues))
    val uDomain = Uniform(low=domain._1, high=domain._2)
    val normal01 = Gaussian(0, scala.math.sqrt(sigma2))
    val (lo, hi) = sparsity
    val regularX = linspace(domain._1, domain._2, nRegular)
    val (lX, lY)= List.fill(nSubjects)({
      val n = lo + nextInt(hi-lo)
      val xx: DenseVector[Double] = DenseVector[Double](List.fill(n)(uDomain.draw()):_*)
      val score: DenseVector[Double] = mvn.draw()
      val yy: DenseVector[Double] = xx.mapValues(i => meanFunction(i) + eigenFunctions(i).t * score + normal01.draw())
      (xx, yy)
    }).unzip
    Fd(lY, lX, regularX)
  }

}
