"""Separate manufacturer corpus and runtime source catalog; no v2 mutation."""
from pathlib import Path
from .config import ROOT, DOCUMENT_SOURCES

V3_INDEX = ROOT/'data/corpus-v3/index'
V3_RAW = ROOT/'.cache/v3/manuals'
SOURCES_V3 = {
    **{k:{**v,'category':category,'series':series,'legacy':True} for k,v,category,series in [
        ('stm32f103',DOCUMENT_SOURCES['stm32f103'],'MCU','stm32f1'),
        ('rp2040',DOCUMENT_SOURCES['rp2040'],'MCU','rp2040'),
        ('tmp117',DOCUMENT_SOURCES['tmp117'],'environment','tmp11x'),
        ('ina226',DOCUMENT_SOURCES['ina226'],'current_monitor','ina22x'),
        ('bme280',DOCUMENT_SOURCES['bme280'],'environment','bme280')]},
    'msp430g2553':{'title':'MSP430G2x53/G2x13 mixed signal MCU datasheet','url':'https://www.ti.com/lit/ds/symlink/msp430g2553.pdf','category':'MCU','series':'msp430g2'},
    'ads1115':{'title':'ADS1113/ADS1114/ADS1115 ADC datasheet','url':'https://www.ti.com/lit/ds/symlink/ads1115.pdf','category':'ADC','series':'ads111x'},
    'ads1220':{'title':'ADS1220 24-bit ADC datasheet','url':'https://www.ti.com/lit/ds/symlink/ads1220.pdf','category':'ADC','series':'ads122x'},
    'ads1256':{'title':'ADS1255/ADS1256 ADC datasheet','url':'https://www.ti.com/lit/ds/symlink/ads1256.pdf','category':'ADC','series':'ads125x'},
    'ina219':{'title':'INA219 current and power monitor datasheet','url':'https://www.ti.com/lit/ds/symlink/ina219.pdf','category':'current_monitor','series':'ina21x'},
    'ina228':{'title':'INA228 20-bit current voltage power monitor datasheet','url':'https://www.ti.com/lit/ds/symlink/ina228.pdf','category':'current_monitor','series':'ina22x'},
    'hdc1080':{'title':'HDC1080 humidity and temperature sensor datasheet','url':'https://www.ti.com/lit/ds/symlink/hdc1080.pdf','category':'environment','series':'hdc10xx'},
    'hdc3020':{'title':'HDC3020/HDC3021/HDC3022 humidity sensor datasheet','url':'https://www.ti.com/lit/ds/symlink/hdc3020.pdf','category':'environment','series':'hdc30xx'},
    'opt3001':{'title':'OPT3001 ambient light sensor datasheet','url':'https://www.ti.com/lit/ds/symlink/opt3001.pdf','category':'environment','series':'opt300x'},
    'bmi160':{'title':'BMI160 low power inertial measurement unit datasheet','url':'https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi160-ds000.pdf','category':'motion','series':'bmi16x'},
    'bmi270':{'title':'BMI270 inertial measurement unit datasheet','url':'https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi270-ds000.pdf','category':'motion','series':'bmi27x'},
    'sn65hvd230':{'title':'SN65HVD230/231/232 CAN transceiver datasheet','url':'https://www.ti.com/lit/ds/symlink/sn65hvd230.pdf','category':'interface','series':'sn65hvd23x'},
    'max3232':{'title':'MAX3232 RS232 transceiver datasheet','url':'https://www.ti.com/lit/ds/symlink/max3232.pdf','category':'interface','series':'max3232'},
    'tca9548a':{'title':'TCA9548A I2C switch datasheet','url':'https://www.ti.com/lit/ds/symlink/tca9548a.pdf','category':'interface','series':'tca954x'},
    'dac80501':{'title':'DAC80501/70501/60501 DAC datasheet','url':'https://www.ti.com/lit/ds/symlink/dac80501.pdf','category':'DAC','series':'dacx0501'},
}

def paths(corpus):
    if corpus=='v3':return V3_INDEX,V3_RAW,SOURCES_V3
    if corpus=='v2':
        from .config import EXPANDED_INDEX,DATA
        return EXPANDED_INDEX,DATA/'raw',DOCUMENT_SOURCES
    raise ValueError('Unknown corpus version')
