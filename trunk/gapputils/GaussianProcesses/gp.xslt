<?xml version="1.0" encoding="utf-8"?>

<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                              xmlns:fn="http://www.w3.org/2005/xpath-functions">
  <xsl:output method="text"/>

  <xsl:template match="/GaussianProcesses-GPTest">
  
  \documentclass{article}
  
  \usepackage{tikz}
  \usepackage{pgfplots}
  
  \begin{document}
  
  \begin{tikzpicture}
  \begin{axis}
  
  \addplot[draw=gray!50] coordinates {
  <xsl:for-each select="Xstar/Collection/Item">
  <xsl:variable name="pos" select="fn:position()" />
  (<xsl:value-of select="@value" />, <xsl:value-of select="/GaussianProcesses-GPTest/Mu/Collection/Item[$pos]/@value - /GaussianProcesses-GPTest/CI/Collection/Item[$pos]/@value" />)
  </xsl:for-each>
  };
  
  \addplot[draw=gray!50] coordinates {
  <xsl:for-each select="Xstar/Collection/Item">
  <xsl:variable name="pos" select="fn:position()" />
  (<xsl:value-of select="@value" />, <xsl:value-of select="/GaussianProcesses-GPTest/Mu/Collection/Item[$pos]/@value + /GaussianProcesses-GPTest/CI/Collection/Item[$pos]/@value" />)
  </xsl:for-each>
  };
  
  \addplot[blue] coordinates {
  <xsl:for-each select="Xstar/Collection/Item">
  <xsl:variable name="pos" select="fn:position()" />
  (<xsl:value-of select="@value" />, <xsl:value-of select="/GaussianProcesses-GPTest/Mu/Collection/Item[$pos]/@value" />)
  </xsl:for-each>
  };
  
  \addplot[red, mark=x, only marks] coordinates {
  <xsl:for-each select="X/Collection/Item">
  <xsl:variable name="pos" select="fn:position()" />
  (<xsl:value-of select="@value" />, <xsl:value-of select="/GaussianProcesses-GPTest/Y/Collection/Item[$pos]/@value" />)
  </xsl:for-each>
  };
  
  \end{axis}
  \end{tikzpicture}
  
  \end{document}
  
  </xsl:template>
 </xsl:stylesheet>
