app/src/main/java/com/example/weather_app/ui/main/MainViewModel.kt
package com.example.weather_app.ui.main

import android.app.Application
import android.widget.Toast
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.weather_app.R
import com.example.weather_app.repository.Repository
import com.example.weather_app.utils.Resource
import kotlinx.coroutines.*
import java.lang.Exception

class MainViewModel(application: Application, private val repository: Repository) : AndroidViewModel(application) {

    private val _weather = MutableLiveData<Resource<MainWeatherResponse>>()
    val weather get() = _weather

    private val _error = MutableLiveData<String>()
    val error get() = _error

    private val _city = MutableLiveData<String>()
    val city get() = _city

    val isLoading = MutableLiveData(false)

    init {
        viewModelScope.launch {
            try {
                isLoading.postValue(true)
                _city.postValue(repository.getLocationNameByIp())
                _weather.postValue(Resource.Success(repository.getWeather(_city.value!!)))
                isLoading.postValue(false)
            } catch (e: Exception) {
                isLoading.postValue(false)
                _error.postValue(getApplication<Application>().getString(R.string.no_connection_message))
            }
        }
    }

    fun getLocationNameByIp(){
        viewModelScope.launch {
            try {
                isLoading.postValue(true)
                _city.postValue(repository.getLocationNameByIp())
                _weather.postValue(Resource.Success(repository.getWeather(_city.value!!)))
                isLoading.postValue(false)
            } catch (e: Exception) {
                isLoading.postValue(false)
                _error.postValue(getApplication<Application>().getString(R.string.no_connection_message))
            }
        }
    }

    class MainViewModelFactory(private val application: Application, private val repository: Repository) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
                @Suppress("UNCHECKED_CAST")
                return MainViewModel(application, repository) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class")
        }
    }
}
