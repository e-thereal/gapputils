<?xml version="1.0" encoding="utf-8"?>

<xsl:stylesheet version="2.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                              xmlns:fn="http://www.w3.org/2005/xpath-functions">
  <xsl:output method="text"/>

  <xsl:template match="/gapputils-InterfaceDescription">  
#include &lt;gapputils/WorkflowInterface.h&gt;

#include &lt;capputils/EnumerableAttribute.h&gt;
#include &lt;capputils/FileExists.h&gt;
#include &lt;capputils/FilenameAttribute.h&gt;
#include &lt;capputils/InputAttribute.h&gt;
#include &lt;capputils/OutputAttribute.h&gt;
#include &lt;capputils/ObserveAttribute.h&gt;
#include &lt;capputils/TimeStampAttribute.h&gt;
#include &lt;capputils/VolatileAttribute.h&gt;

#include &lt;capputils/FromEnumerableAttribute.h&gt;
#include &lt;capputils/ToEnumerableAttribute.h&gt;

#include &lt;gapputils/HideAttribute.h&gt;
#include &lt;gapputils/ReadOnlyAttribute.h&gt;

<xsl:for-each select="Headers/Collection/Item">#include &lt;<xsl:value-of select="@value" />&gt;
</xsl:for-each>
namespace gapputils {
  
namespace host {
  
namespace internal {
  
/*** Declaration ***/
  
class <xsl:value-of select="Name/@value" /> : public gapputils::workflow::WorkflowInterface
{
  InitReflectableClass(<xsl:value-of select="Name/@value" />)
  
  <xsl:for-each select="PropertyDescriptions/Collection/Item/gapputils-PropertyDescription">Property(<xsl:value-of select="Name/@value" />, <xsl:value-of select="Type/@value" />)
  </xsl:for-each>
private:
  int _dummy;
  static int <xsl:for-each select="PropertyDescriptions/Collection/Item/gapputils-PropertyDescription"><xsl:value-of select="Name/@value" />Id, </xsl:for-each>_dummyId;
  
public:
  <xsl:value-of select="Name/@value" />(void);
  virtual ~<xsl:value-of select="Name/@value" />(void);
};

/*** Implementation ***/

<xsl:for-each select="PropertyDescriptions/Collection/Item/gapputils-PropertyDescription">int <xsl:value-of select="/gapputils-InterfaceDescription/Name/@value" />::<xsl:value-of select="Name/@value" />Id;
</xsl:for-each>
int <xsl:value-of select="/gapputils-InterfaceDescription/Name/@value" />::_dummyId;

BeginPropertyDefinitions(<xsl:value-of select="Name/@value" />)
  using namespace capputils::attributes;
  using namespace gapputils::attributes;
 
  ReflectableBase(gapputils::workflow::WorkflowInterface)
  <xsl:for-each select="PropertyDescriptions/Collection/Item/gapputils-PropertyDescription">DefineProperty(<xsl:value-of select="Name/@value" />,
    <xsl:for-each select="PropertyAttributes/Collection/Item"><xsl:value-of select="@value" />,
    </xsl:for-each>Observe(<xsl:value-of select="Name/@value" />Id = PROPERTY_ID))
  </xsl:for-each>
EndPropertyDefinitions

<xsl:value-of select="Name/@value" />::<xsl:value-of select="Name/@value" />() :
  _dummy(0)<xsl:for-each select="PropertyDescriptions/Collection/Item/gapputils-PropertyDescription[DefaultValue]">,
  _<xsl:value-of select="Name/@value" />(<xsl:value-of select="DefaultValue/@value" />)</xsl:for-each>
{
  setLabel("New Interface");
}

<xsl:value-of select="Name/@value" />::~<xsl:value-of select="Name/@value" />() { }
  
}
  
}
  
}
  
  </xsl:template>
 </xsl:stylesheet>
