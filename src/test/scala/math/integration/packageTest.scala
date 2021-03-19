package fda
package math.integration

import breeze.linalg.linspace
import org.scalatest.FunSuite

class packageTest extends FunSuite {

  test("testRombergIntegrator") {
    val f: Double => Double = x => scala.math.pow(x, 2)
    val k = 13
    val J = scala.math.pow(2, k)
    val x = linspace(0.0, 10.0, J.toInt + 1)
    val fx = x.mapValues(f)
    val I = rombergIntegrator(fx, k, 10.0 / J)
    assert(I == scala.math.pow(10, 3) / 3.0)
  }

}
