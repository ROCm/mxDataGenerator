virtual bool isOne(uint8_t const* scaleBytes,
                   uint8_t const* dataBytes,
                   size_t         scaleIndex,
                   size_t         dataIndex) const override;

virtual bool isZero(uint8_t const* scaleBytes,
                    uint8_t const* dataBytes,
                    size_t         scaleIndex,
                    size_t         dataIndex) const override;

virtual bool isNaN(uint8_t const* scaleBytes,
                   uint8_t const* dataBytes,
                   size_t         scaleIndex,
                   size_t         dataIndex) const override;

virtual bool isInf(uint8_t const* scaleBytes,
                   uint8_t const* dataBytes,
                   size_t         scaleIndex,
                   size_t         dataIndex) const override;

virtual bool isLess(double         val,
                    uint8_t const* scaleBytes,
                    uint8_t const* dataBytes,
                    size_t         scaleIndex,
                    size_t         dataInd) const override;

virtual bool isGreater(double         val,
                       uint8_t const* scaleBytes,
                       uint8_t const* dataBytes,
                       size_t         scaleIndex,
                       size_t         dataInd) const override;

virtual double toDouble(uint8_t const* scaleBytes,
                        uint8_t const* dataBytes,
                        size_t         scaleIndex,
                        size_t         dataInde) const override;

virtual float toFloat(uint8_t const* scaleBytes,
                      uint8_t const* dataBytes,
                      size_t         scaleIndex,
                      size_t         dataIndex) const override;

virtual bool isOnePacked(uint8_t const* scaleBytes,
                         uint8_t const* dataBytes,
                         size_t         scaleIndex,
                         size_t         dataIn) const override;

virtual bool isZeroPacked(uint8_t const* scaleBytes,
                          uint8_t const* dataBytes,
                          size_t         scaleIndex,
                          size_t         dataIndex) const override;

virtual bool isNaNPacked(uint8_t const* scaleBytes,
                         uint8_t const* dataBytes,
                         size_t         scaleIndex,
                         size_t         dataIndex) const override;

virtual bool isInfPacked(uint8_t const* scaleBytes,
                         uint8_t const* dataBytes,
                         size_t         scaleIndex,
                         size_t         dataIndex) const override;

virtual bool isLessPacked(double         val,
                          uint8_t const* scaleBytes,
                          uint8_t const* dataBytes,
                          size_t         scaleIndex,
                          size_t         dataIn) const override;

virtual bool isGreaterPacked(double         val,
                             uint8_t const* scaleBytes,
                             uint8_t const* dataBytes,
                             size_t         scaleIndex,
                             size_t         dataInd) const override;

virtual bool isSubnorm(uint8_t const* dataBytes, size_t dataIndex) const override;

virtual bool isSubnormPacked(uint8_t const* dataBytes, size_t dataIndex) const override;

virtual double toDoublePacked(uint8_t const* scaleBytes,
                              uint8_t const* dataBytes,
                              size_t         scaleIndex,
                              size_t         dataInd) const override;

virtual float toFloatPacked(uint8_t const* scaleBytes,
                            uint8_t const* dataBytes,
                            size_t         scaleIndex,
                            size_t         dataIndex) const override;

virtual void setOne(uint8_t* scaleBytes,
                    uint8_t* dataBytes,
                    size_t   scaleIndex,
                    size_t   dataIndex,
                    bool     subNormal = false) const override;

virtual void setZero(uint8_t* scaleBytes,
                     uint8_t* dataBytes,
                     size_t   scaleIndex,
                     size_t   dataIndex) const override;

virtual void setNaN(uint8_t* scaleBytes,
                    uint8_t* dataBytes,
                    size_t   scaleIndex,
                    size_t   dataIndex) const override;

virtual void setInf(uint8_t* scaleBytes,
                    uint8_t* dataBytes,
                    size_t   scaleIndex,
                    size_t   dataIndex) const override;

virtual void setDataMax(uint8_t* dataBytes,
                        size_t   dataIndex,
                        bool     subNormal = false,
                        bool     positive  = true) const override;

virtual void setOnePacked(uint8_t* scaleBytes,
                          uint8_t* dataBytes,
                          size_t   scaleIndex,
                          size_t   dataIndex,
                          bool     subNormal = false) const override;

virtual void setZeroPacked(uint8_t* scaleBytes,
                           uint8_t* dataBytes,
                           size_t   scaleIndex,
                           size_t   dataIndex) const override;

virtual void setNaNPacked(uint8_t* scaleBytes,
                          uint8_t* dataBytes,
                          size_t   scaleIndex,
                          size_t   dataIndex) const override;

virtual void setInfPacked(uint8_t* scaleBytes,
                          uint8_t* dataBytes,
                          size_t   scaleIndex,
                          size_t   dataIndex) const override;

virtual void setDataMaxPacked(uint8_t* dataBytes,
                              size_t   dataIndex,
                              bool     subNormal = false,
                              bool     positive  = true) const override;

virtual uint64_t satConvertToType(float value) const override;

virtual uint64_t nonSatConvertToType(float value) const override;
